#include "server-slot-sched.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & ckpt) {
            const auto & cur = *ckpt;
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = **it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = *it;

    return res;
}

//
// server_prompt_cache
//
server_prompt_data::~server_prompt_data() {
    clear();
}

server_prompt_data::server_prompt_data(server_prompt_data && other) noexcept {
    *this = std::move(other);
}

server_prompt_data & server_prompt_data::operator=(server_prompt_data && other) noexcept {
    if (this == &other) {
        return *this;
    }

    clear();

    main         = std::move(other.main);
    drft         = std::move(other.drft);
    cache_file   = std::move(other.cache_file);
    mapping      = other.mapping;
    mapping_size = other.mapping_size;
    main_size    = other.main_size;
    drft_size    = other.drft_size;
    remove_file_on_destroy = other.remove_file_on_destroy;

#ifdef _WIN32
    file_handle    = other.file_handle;
    mapping_handle = other.mapping_handle;
    other.file_handle    = nullptr;
    other.mapping_handle = nullptr;
#endif

    other.mapping      = nullptr;
    other.mapping_size = 0;
    other.main_size    = 0;
    other.drft_size    = 0;
    other.remove_file_on_destroy = true;
    other.cache_file.clear();

    return *this;
}

void server_prompt_data::clear() {
    if (mapping != nullptr) {
#ifdef _WIN32
        UnmapViewOfFile(mapping);
#else
        munmap(mapping, mapping_size);
#endif
        mapping = nullptr;
    }

#ifdef _WIN32
    if (mapping_handle != nullptr) {
        CloseHandle((HANDLE) mapping_handle);
        mapping_handle = nullptr;
    }
    if (file_handle != nullptr) {
        CloseHandle((HANDLE) file_handle);
        file_handle = nullptr;
    }
#endif

    if (!cache_file.empty()) {
        if (remove_file_on_destroy) {
            std::error_code ec;
            std::filesystem::remove(cache_file, ec);
            if (ec) {
                SRV_WRN("failed to remove disk cache file %s: %s\n", cache_file.c_str(), ec.message().c_str());
            }

            ec.clear();
            const std::string metadata_file = cache_file + ".meta";
            std::filesystem::remove(metadata_file, ec);
            if (ec) {
                SRV_WRN("failed to remove disk cache metadata %s: %s\n", metadata_file.c_str(), ec.message().c_str());
            }
        }
        cache_file.clear();
    }

    mapping_size = 0;
    main_size = 0;
    drft_size = 0;
    remove_file_on_destroy = true;
    main.clear();
    drft.clear();
}

void server_prompt_data::discard() {
    remove_file_on_destroy = true;
    clear();
}

static bool server_prompt_cache_alloc_disk(
        server_prompt_data & data,
        const std::string & cache_dir,
        uint64_t & next_file_id,
        size_t size) {
    if (size == 0) {
        SRV_ERR("%s", "cannot allocate an empty disk cache state\n");
        return false;
    }

    if (next_file_id == 0) {
        next_file_id = (uint64_t) ggml_time_us();
    }

    for (;;) {
        const std::filesystem::path path = std::filesystem::path(cache_dir) /
            ("llama-prompt-cache-" + std::to_string(next_file_id++) + ".bin");
        const std::string path_str = path.string();

#ifdef _WIN32
        HANDLE file = CreateFileA(
            path_str.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
            CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            if (GetLastError() == ERROR_FILE_EXISTS) {
                continue;
            }
            SRV_ERR("failed to create disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
            return false;
        }

        LARGE_INTEGER file_size;
        file_size.QuadPart = size;
        if (!SetFilePointerEx(file, file_size, nullptr, FILE_BEGIN) || !SetEndOfFile(file)) {
            SRV_ERR("failed to resize disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
            CloseHandle(file);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        HANDLE mapping_handle = CreateFileMappingA(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (mapping_handle == nullptr) {
            SRV_ERR("failed to map disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
            CloseHandle(file);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        void * mapping = MapViewOfFile(mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (mapping == nullptr) {
            SRV_ERR("failed to open disk cache mapping %s (error %lu)\n", path_str.c_str(), GetLastError());
            CloseHandle(mapping_handle);
            CloseHandle(file);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        data.cache_file    = path_str;
        data.mapping       = (uint8_t *) mapping;
        data.mapping_size  = size;
        data.file_handle   = file;
        data.mapping_handle = mapping_handle;
#else
        if (size > (size_t) std::numeric_limits<off_t>::max()) {
            SRV_ERR("disk cache state is too large: %zu bytes\n", size);
            return false;
        }

        const int fd = open(path_str.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            if (errno == EEXIST) {
                continue;
            }
            SRV_ERR("failed to create disk cache file %s: %s\n", path_str.c_str(), strerror(errno));
            return false;
        }

        if (ftruncate(fd, (off_t) size) != 0) {
            SRV_ERR("failed to resize disk cache file %s: %s\n", path_str.c_str(), strerror(errno));
            close(fd);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        void * mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int mmap_errno = errno;
        close(fd);
        if (mapping == MAP_FAILED) {
            SRV_ERR("failed to map disk cache file %s: %s\n", path_str.c_str(), strerror(mmap_errno));
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        data.cache_file   = path_str;
        data.mapping      = (uint8_t *) mapping;
        data.mapping_size = size;
#endif

        return true;
    }
}

size_t server_prompt_state_get_staged(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    std::vector<uint8_t> staging;
    try {
        staging.resize(size);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate %zu bytes to stage prompt state: %s\n", size, e.what());
        return 0;
    }
    const size_t n = llama_state_seq_get_data_ext(ctx, staging.data(), size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n != size) {
        return n;
    }
    memcpy(dst, staging.data(), size);
    return n;
}

size_t server_prompt_state_set_staged(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    std::vector<uint8_t> staging;
    try {
        staging.assign(src, src + size);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate %zu bytes to stage prompt state: %s\n", size, e.what());
        return 0;
    }
    return llama_state_seq_set_data_ext(ctx, staging.data(), size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
}

namespace {

constexpr char SERVER_PROMPT_CACHE_META_MAGIC[8] = {'L', 'L', 'P', 'C', 'A', 'C', 'H', 'E'};
constexpr uint32_t SERVER_PROMPT_CACHE_META_VERSION = 1;
constexpr uint32_t SERVER_PROMPT_CACHE_META_FLAG_MTMD = 1u << 0;
constexpr uint64_t SERVER_PROMPT_CACHE_META_HEADER_SIZE = 64;
constexpr uint64_t SERVER_PROMPT_CACHE_META_CHECKPOINT_SIZE = 80;

template <typename T>
bool server_prompt_cache_meta_write(std::ostream & output, const T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "cache metadata values must be trivially copyable");
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return output.good();
}

template <typename T>
bool server_prompt_cache_meta_read(std::istream & input, T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "cache metadata values must be trivially copyable");
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return input.good();
}

bool server_prompt_cache_u64_add(uint64_t & value, uint64_t addend) {
    if (addend > std::numeric_limits<uint64_t>::max() - value) {
        return false;
    }
    value += addend;
    return true;
}

bool server_prompt_cache_u64_mul(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool server_prompt_cache_range_valid(uint64_t offset, uint64_t size, uint64_t total) {
    return offset <= total && size <= total - offset;
}

void server_prompt_cache_remove_files(const std::filesystem::path & data_path) {
    std::error_code ec;
    std::filesystem::remove(data_path, ec);
    if (ec) {
        SRV_WRN("failed to remove disk cache file %s: %s\n", data_path.string().c_str(), ec.message().c_str());
    }

    ec.clear();
    const std::filesystem::path metadata_path = data_path.string() + ".meta";
    std::filesystem::remove(metadata_path, ec);
    if (ec) {
        SRV_WRN("failed to remove disk cache metadata %s: %s\n", metadata_path.string().c_str(), ec.message().c_str());
    }

    ec.clear();
    const std::filesystem::path temporary_path = metadata_path.string() + ".tmp";
    std::filesystem::remove(temporary_path, ec);
    if (ec) {
        SRV_WRN("failed to remove temporary disk cache metadata %s: %s\n", temporary_path.string().c_str(), ec.message().c_str());
    }
}

bool server_prompt_cache_flush_disk(server_prompt_data & data) {
#ifdef _WIN32
    if (!FlushViewOfFile(data.mapping, data.mapping_size)) {
        SRV_ERR("failed to flush disk cache mapping %s (error %lu)\n", data.cache_file.c_str(), GetLastError());
        return false;
    }
    if (!FlushFileBuffers((HANDLE) data.file_handle)) {
        SRV_ERR("failed to flush disk cache file %s (error %lu)\n", data.cache_file.c_str(), GetLastError());
        return false;
    }
#else
    if (msync(data.mapping, data.mapping_size, MS_SYNC) != 0) {
        SRV_ERR("failed to flush disk cache file %s: %s\n", data.cache_file.c_str(), strerror(errno));
        return false;
    }
#endif
    return true;
}

bool server_prompt_cache_open_disk(
        server_prompt_data & data,
        const std::filesystem::path & path,
        size_t size) {
    const std::string path_str = path.string();

#ifdef _WIN32
    HANDLE file = CreateFileA(
        path_str.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        SRV_WRN("failed to open disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
        return false;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0 || (uint64_t) file_size.QuadPart != size) {
        SRV_WRN("disk cache file has an unexpected size: %s\n", path_str.c_str());
        CloseHandle(file);
        return false;
    }

    HANDLE mapping_handle = CreateFileMappingA(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (mapping_handle == nullptr) {
        SRV_WRN("failed to map disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
        CloseHandle(file);
        return false;
    }

    void * mapping = MapViewOfFile(mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (mapping == nullptr) {
        SRV_WRN("failed to open disk cache mapping %s (error %lu)\n", path_str.c_str(), GetLastError());
        CloseHandle(mapping_handle);
        CloseHandle(file);
        return false;
    }

    data.file_handle    = file;
    data.mapping_handle = mapping_handle;
#else
    const int fd = open(path_str.c_str(), O_RDWR);
    if (fd < 0) {
        SRV_WRN("failed to open disk cache file %s: %s\n", path_str.c_str(), strerror(errno));
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0 || (uint64_t) st.st_size != size) {
        SRV_WRN("disk cache file has an unexpected size: %s\n", path_str.c_str());
        close(fd);
        return false;
    }

    void * mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    const int mmap_errno = errno;
    close(fd);
    if (mapping == MAP_FAILED) {
        SRV_WRN("failed to map disk cache file %s: %s\n", path_str.c_str(), strerror(mmap_errno));
        return false;
    }
#endif

    data.cache_file = path_str;
    data.mapping = (uint8_t *) mapping;
    data.mapping_size = size;
    data.remove_file_on_destroy = false;

    return true;
}

bool server_prompt_cache_write_metadata(
        const server_prompt_cache_state & state,
        const std::string & cache_key) {
    const std::vector<char> serialized_tokens = state.prompt.tokens.serialize();

    const std::filesystem::path metadata_path = state.data.cache_file + ".meta";
    const std::filesystem::path temporary_path = metadata_path.string() + ".tmp";

    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        SRV_ERR("failed to create disk cache metadata %s\n", temporary_path.string().c_str());
        return false;
    }

    // the metadata holds the prompt tokens: owner-only, like the state file, before anything is written
    std::error_code ec_perm;
    std::filesystem::permissions(temporary_path,
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace, ec_perm);

    const uint32_t version = SERVER_PROMPT_CACHE_META_VERSION;
    const uint32_t flags = state.prompt.tokens.has_mtmd ? SERVER_PROMPT_CACHE_META_FLAG_MTMD : 0;
    const uint64_t key_size = cache_key.size();
    const uint64_t tokens_size = serialized_tokens.size();
    const uint64_t payload_size = state.data.mapping_size;
    const uint64_t main_size = state.data.main_size;
    const uint64_t drft_size = state.data.drft_size;
    const uint64_t checkpoint_count = state.checkpoints_disk.size();

    output.write(SERVER_PROMPT_CACHE_META_MAGIC, sizeof(SERVER_PROMPT_CACHE_META_MAGIC));
    bool ok = output.good() &&
        server_prompt_cache_meta_write(output, version) &&
        server_prompt_cache_meta_write(output, flags) &&
        server_prompt_cache_meta_write(output, key_size) &&
        server_prompt_cache_meta_write(output, tokens_size) &&
        server_prompt_cache_meta_write(output, payload_size) &&
        server_prompt_cache_meta_write(output, main_size) &&
        server_prompt_cache_meta_write(output, drft_size) &&
        server_prompt_cache_meta_write(output, checkpoint_count);

    if (ok && key_size > 0) {
        output.write(cache_key.data(), key_size);
        ok = output.good();
    }
    if (ok && tokens_size > 0) {
        output.write(serialized_tokens.data(), tokens_size);
        ok = output.good();
    }

    for (const auto & checkpoint : state.checkpoints_disk) {
        const int64_t n_tokens = checkpoint.n_tokens;
        const int32_t id_task = checkpoint.id_task;
        const uint32_t reserved = 0;
        const int64_t pos_min = checkpoint.pos_min;
        const int64_t pos_max = checkpoint.pos_max;
        const uint64_t offset_tgt = checkpoint.offset_tgt;
        const uint64_t size_tgt = checkpoint.size_tgt;
        const uint64_t offset_dft = checkpoint.offset_dft;
        const uint64_t size_dft = checkpoint.size_dft;
        const uint64_t offset_spec = checkpoint.offset_spec;
        const uint64_t size_spec = checkpoint.size_spec;

        ok = ok &&
            server_prompt_cache_meta_write(output, n_tokens) &&
            server_prompt_cache_meta_write(output, id_task) &&
            server_prompt_cache_meta_write(output, reserved) &&
            server_prompt_cache_meta_write(output, pos_min) &&
            server_prompt_cache_meta_write(output, pos_max) &&
            server_prompt_cache_meta_write(output, offset_tgt) &&
            server_prompt_cache_meta_write(output, size_tgt) &&
            server_prompt_cache_meta_write(output, offset_dft) &&
            server_prompt_cache_meta_write(output, size_dft) &&
            server_prompt_cache_meta_write(output, offset_spec) &&
            server_prompt_cache_meta_write(output, size_spec);
    }

    output.close();
    ok = ok && !output.fail();

    if (!ok) {
        SRV_ERR("failed to write disk cache metadata %s\n", temporary_path.string().c_str());
        std::error_code ec;
        std::filesystem::remove(temporary_path, ec);
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(temporary_path, metadata_path, ec);
    if (ec) {
        SRV_ERR("failed to commit disk cache metadata %s: %s\n", metadata_path.string().c_str(), ec.message().c_str());
        ec.clear();
        std::filesystem::remove(temporary_path, ec);
        return false;
    }

    return true;
}

bool server_prompt_cache_read_metadata(
        const std::filesystem::path & metadata_path,
        const std::string & expected_cache_key,
        bool expected_has_mtmd,
        server_prompt_cache_state & state) {
    std::error_code ec;
    const uintmax_t metadata_file_size = std::filesystem::file_size(metadata_path, ec);
    if (ec || metadata_file_size < SERVER_PROMPT_CACHE_META_HEADER_SIZE) {
        return false;
    }

    std::ifstream input(metadata_path, std::ios::binary);
    if (!input) {
        return false;
    }

    char magic[sizeof(SERVER_PROMPT_CACHE_META_MAGIC)];
    input.read(magic, sizeof(magic));

    uint32_t version = 0;
    uint32_t flags = 0;
    uint64_t key_size = 0;
    uint64_t tokens_size = 0;
    uint64_t payload_size = 0;
    uint64_t main_size = 0;
    uint64_t drft_size = 0;
    uint64_t checkpoint_count = 0;

    bool ok = input.good() &&
        server_prompt_cache_meta_read(input, version) &&
        server_prompt_cache_meta_read(input, flags) &&
        server_prompt_cache_meta_read(input, key_size) &&
        server_prompt_cache_meta_read(input, tokens_size) &&
        server_prompt_cache_meta_read(input, payload_size) &&
        server_prompt_cache_meta_read(input, main_size) &&
        server_prompt_cache_meta_read(input, drft_size) &&
        server_prompt_cache_meta_read(input, checkpoint_count);

    uint64_t checkpoint_bytes = 0;
    uint64_t expected_metadata_size = SERVER_PROMPT_CACHE_META_HEADER_SIZE;
    ok = ok && memcmp(magic, SERVER_PROMPT_CACHE_META_MAGIC, sizeof(magic)) == 0 &&
        version == SERVER_PROMPT_CACHE_META_VERSION &&
        (flags & ~SERVER_PROMPT_CACHE_META_FLAG_MTMD) == 0 &&
        ((flags & SERVER_PROMPT_CACHE_META_FLAG_MTMD) != 0) == expected_has_mtmd &&
        key_size <= 64 * 1024 &&
        tokens_size > 0 &&
        tokens_size <= 1024ull * 1024ull * 1024ull &&
        tokens_size <= std::numeric_limits<size_t>::max() &&
        tokens_size % sizeof(llama_token) == 0 &&
        payload_size > 0 &&
        payload_size <= std::numeric_limits<size_t>::max() &&
        main_size <= std::numeric_limits<size_t>::max() &&
        drft_size <= std::numeric_limits<size_t>::max() &&
        checkpoint_count <= 1024 * 1024 &&
        checkpoint_count <= std::numeric_limits<size_t>::max() &&
        server_prompt_cache_u64_mul(checkpoint_count, SERVER_PROMPT_CACHE_META_CHECKPOINT_SIZE, checkpoint_bytes) &&
        server_prompt_cache_u64_add(expected_metadata_size, key_size) &&
        server_prompt_cache_u64_add(expected_metadata_size, tokens_size) &&
        server_prompt_cache_u64_add(expected_metadata_size, checkpoint_bytes) &&
        expected_metadata_size == metadata_file_size &&
        main_size <= payload_size &&
        drft_size <= payload_size - main_size;
    if (!ok) {
        return false;
    }

    std::string cache_key(key_size, '\0');
    if (key_size > 0) {
        input.read(cache_key.data(), key_size);
    }
    if (!input.good() || cache_key != expected_cache_key) {
        return false;
    }

    std::vector<char> serialized_tokens(tokens_size);
    input.read(serialized_tokens.data(), tokens_size);
    if (!input.good()) {
        return false;
    }

    llama_tokens packed_tokens(tokens_size / sizeof(llama_token));
    memcpy(packed_tokens.data(), serialized_tokens.data(), tokens_size);
    try {
        state.prompt.tokens = server_tokens::deserialize(packed_tokens, expected_has_mtmd);
    } catch (const std::exception & e) {
        SRV_WRN("failed to deserialize tokens from disk cache metadata %s: %s\n", metadata_path.string().c_str(), e.what());
        return false;
    }
    if (state.prompt.tokens.empty()) {
        return false;
    }

    state.data.main_size = main_size;
    state.data.drft_size = drft_size;
    state.checkpoints_disk.reserve(checkpoint_count);

    uint64_t next_offset = main_size + drft_size;
    for (uint64_t i = 0; i < checkpoint_count; ++i) {
        int64_t n_tokens = 0;
        int32_t id_task = -1;
        uint32_t reserved = 0;
        int64_t pos_min = 0;
        int64_t pos_max = 0;
        uint64_t offset_tgt = 0;
        uint64_t size_tgt = 0;
        uint64_t offset_dft = 0;
        uint64_t size_dft = 0;
        uint64_t offset_spec = 0;
        uint64_t size_spec = 0;

        ok =
            server_prompt_cache_meta_read(input, n_tokens) &&
            server_prompt_cache_meta_read(input, id_task) &&
            server_prompt_cache_meta_read(input, reserved) &&
            server_prompt_cache_meta_read(input, pos_min) &&
            server_prompt_cache_meta_read(input, pos_max) &&
            server_prompt_cache_meta_read(input, offset_tgt) &&
            server_prompt_cache_meta_read(input, size_tgt) &&
            server_prompt_cache_meta_read(input, offset_dft) &&
            server_prompt_cache_meta_read(input, size_dft) &&
            server_prompt_cache_meta_read(input, offset_spec) &&
            server_prompt_cache_meta_read(input, size_spec);

        ok = ok && reserved == 0 &&
            n_tokens >= 0 && (uint64_t) n_tokens <= state.prompt.tokens.size() &&
            pos_min >= std::numeric_limits<llama_pos>::min() && pos_min <= std::numeric_limits<llama_pos>::max() &&
            pos_max >= std::numeric_limits<llama_pos>::min() && pos_max <= std::numeric_limits<llama_pos>::max() &&
            offset_tgt == next_offset && server_prompt_cache_range_valid(offset_tgt, size_tgt, payload_size);
        if (!ok || !server_prompt_cache_u64_add(next_offset, size_tgt)) {
            return false;
        }
        ok = offset_dft == next_offset && server_prompt_cache_range_valid(offset_dft, size_dft, payload_size);
        if (!ok || !server_prompt_cache_u64_add(next_offset, size_dft)) {
            return false;
        }
        ok = offset_spec == next_offset && server_prompt_cache_range_valid(offset_spec, size_spec, payload_size);
        if (!ok || !server_prompt_cache_u64_add(next_offset, size_spec)) {
            return false;
        }

        state.checkpoints_disk.push_back({
            n_tokens,
            id_task,
            (llama_pos) pos_min,
            (llama_pos) pos_max,
            (size_t) offset_tgt,
            (size_t) size_tgt,
            (size_t) offset_dft,
            (size_t) size_dft,
            (size_t) offset_spec,
            (size_t) size_spec,
        });
    }

    if (!input.good() || next_offset != payload_size) {
        return false;
    }

    std::filesystem::path data_path = metadata_path;
    data_path.replace_extension();
    if (!server_prompt_cache_open_disk(state.data, data_path, payload_size)) {
        return false;
    }

    return true;
}

} // namespace

server_prompt_cache::server_prompt_cache(
    int32_t limit_size_mib,
    size_t limit_tokens,
    const std::string & cache_dir,
    const std::string & cache_key,
    bool has_mtmd) :
    limit_size(1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib)),
    limit_tokens(limit_tokens),
    cache_dir(cache_dir),
    cache_key(cache_key),
    has_mtmd(has_mtmd) {
    if (cache_dir.empty()) {
        return;
    }

    std::vector<std::filesystem::path> metadata_paths;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(cache_dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }

        const std::string filename = entry.path().filename().string();
        const bool is_cache_file = filename.rfind("llama-prompt-cache-", 0) == 0;
        if (!is_cache_file) {
            continue;
        }

        if (string_ends_with(filename, ".bin.meta")) {
            // entries written by older builds were world-readable
            std::error_code ec_perm;
            std::filesystem::permissions(entry.path(),
                    std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                    std::filesystem::perm_options::replace, ec_perm);
            metadata_paths.push_back(entry.path());
        } else if (string_ends_with(filename, ".bin.meta.tmp")) {
            std::filesystem::remove(entry.path(), ec);
            ec.clear();
        }
    }
    if (ec) {
        SRV_WRN("failed to scan prompt cache directory %s: %s\n", cache_dir.c_str(), ec.message().c_str());
        return;
    }

    std::sort(metadata_paths.begin(), metadata_paths.end(), [](const auto & lhs, const auto & rhs) {
        std::error_code lhs_ec;
        std::error_code rhs_ec;
        const auto lhs_time = std::filesystem::last_write_time(lhs, lhs_ec);
        const auto rhs_time = std::filesystem::last_write_time(rhs, rhs_ec);
        if (lhs_ec || rhs_ec) {
            return lhs.string() < rhs.string();
        }
        return lhs_time < rhs_time;
    });

    for (const auto & metadata_path : metadata_paths) {
        server_prompt_cache_state state;
        bool loaded = false;
        try {
            loaded = server_prompt_cache_read_metadata(metadata_path, cache_key, has_mtmd, state);
        } catch (const std::exception & e) {
            SRV_WRN("failed to read disk prompt cache metadata %s: %s\n", metadata_path.string().c_str(), e.what());
        }
        if (!loaded) {
            std::filesystem::path data_path = metadata_path;
            data_path.replace_extension();
            SRV_WRN("removing incompatible or invalid disk prompt cache entry %s\n", data_path.string().c_str());
            server_prompt_cache_remove_files(data_path);
            continue;
        }
        state.id = ++id_last;
        states.push_back(std::move(state));
    }

    // Raw files without a committed metadata sidecar are interrupted writes.
    for (const auto & entry : std::filesystem::directory_iterator(cache_dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        const std::string filename = entry.path().filename().string();
        if (filename.rfind("llama-prompt-cache-", 0) != 0 || !string_ends_with(filename, ".bin")) {
            continue;
        }
        const std::filesystem::path metadata_path = entry.path().string() + ".meta";
        if (!std::filesystem::exists(metadata_path, ec) || ec) {
            ec.clear();
            SRV_WRN("removing incomplete disk prompt cache entry %s\n", entry.path().string().c_str());
            server_prompt_cache_remove_files(entry.path());
        }
    }

    const size_t restored = states.size();
    update();
    SRV_INF("restored %zu persistent prompt cache entries (%.3f MiB) from %s\n",
            states.size(), size() / (1024.0 * 1024.0), cache_dir.c_str());
    if (states.size() < restored) {
        SRV_INF("evicted %zu restored prompt cache entries to satisfy configured limits\n", restored - states.size());
    }
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt->size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it->data.discard();
            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            states.front().data.discard();
            states.pop_front();
        }
    }

    server_prompt_cache_state state;
    state.prompt.tokens = prompt.tokens.clone();

    if (!cache_dir.empty()) {
        if (!server_prompt_cache_alloc_disk(state.data, cache_dir, next_file_id, state_size_new)) {
            return nullptr;
        }

        state.data.main_size = state_size_tgt;
        state.data.drft_size = state_size_dft;

        size_t offset = state_size_tgt + state_size_dft;
        state.checkpoints_disk.reserve(prompt.checkpoints.size());
        for (const auto & ckpt : prompt.checkpoints) {
            const common_prompt_checkpoint & checkpoint = *ckpt;
            server_prompt_cache_checkpoint checkpoint_disk;
            checkpoint_disk.n_tokens = checkpoint.n_tokens;
            checkpoint_disk.id_task  = checkpoint.id_task;
            checkpoint_disk.pos_min  = checkpoint.pos_min;
            checkpoint_disk.pos_max  = checkpoint.pos_max;

            checkpoint_disk.offset_tgt = offset;
            checkpoint_disk.size_tgt   = checkpoint.size_tgt();
            if (checkpoint_disk.size_tgt > 0) {
                if (!checkpoint.copy_tgt(state.data.mapping + offset)) {
                    SRV_WRN("%s", " - failed to read a context checkpoint back from its file, skipping\n");
                    state.data.discard();
                    return nullptr;
                }
                offset += checkpoint_disk.size_tgt;
            }

            checkpoint_disk.offset_dft = offset;
            checkpoint_disk.size_dft   = checkpoint.size_dft();
            if (checkpoint_disk.size_dft > 0) {
                if (!checkpoint.copy_dft(state.data.mapping + offset)) {
                    SRV_WRN("%s", " - failed to read a context checkpoint back from its file, skipping\n");
                    state.data.discard();
                    return nullptr;
                }
                offset += checkpoint_disk.size_dft;
            }

            checkpoint_disk.offset_spec = offset;
            checkpoint_disk.size_spec   = checkpoint.data_spec.size();
            if (checkpoint_disk.size_spec > 0) {
                memcpy(state.data.mapping + offset, checkpoint.data_spec.data(), checkpoint_disk.size_spec);
                offset += checkpoint_disk.size_spec;
            }

            state.checkpoints_disk.push_back(checkpoint_disk);
        }
        GGML_ASSERT(offset == state_size_new);
    } else {
        state.prompt.checkpoints = prompt.checkpoints;

        try {
            state.data.main.resize(state_size_tgt);
            state.data.drft.resize(state_size_dft);
        } catch (const std::bad_alloc & e) {
            SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

            limit_size = std::max<size_t>(1, 0.4*size());

            SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

            update();

            return nullptr;
        }
    }

    state.id = ++id_last;
    states.push_back(std::move(state));

    return &states.back();
}

bool server_prompt_cache::commit(server_prompt_cache_state * state) {
    if (state == nullptr || !state->data.is_disk()) {
        return state != nullptr;
    }

    auto it = std::find_if(states.begin(), states.end(), [state](const auto & candidate) {
        return &candidate == state;
    });
    if (it == states.end()) {
        return false;
    }

    bool ok = server_prompt_cache_flush_disk(it->data);
    try {
        ok = ok && server_prompt_cache_write_metadata(*it, cache_key);
    } catch (const std::exception & e) {
        SRV_ERR("failed to serialize disk prompt cache metadata: %s\n", e.what());
        ok = false;
    }

    if (!ok) {
        it->data.discard();
        states.erase(it);
        return false;
    }

    it->data.remove_file_on_destroy = false;
    return true;
}

void server_prompt_cache::discard(server_prompt_cache_state * state) {
    if (state == nullptr) {
        return;
    }

    auto it = std::find_if(states.begin(), states.end(), [state](const auto & candidate) {
        return &candidate == state;
    });
    if (it == states.end()) {
        return;
    }

    it->data.discard();
    states.erase(it);
}

// the same threshold as server_prompt::get_reuse(), for cache entries whose checkpoints are on disk
llama_pos server_prompt_pos_min_thold(llama_pos pos_next, int32_t n_swa, bool has_new_tokens) {
    const llama_pos pos_eval = pos_next - (has_new_tokens ? 0 : 1);
    return std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);
}

int server_prompt_checkpoint_reuse(
        const server_tokens & tokens,
        int64_t               n_tokens,
        llama_pos             checkpoint_pos_min,
        llama_pos             checkpoint_pos_max,
        llama_pos             pos_next,
        llama_pos             pos_min_thold) {
    if (checkpoint_pos_max > pos_next) {
        return -1;
    }

    if (checkpoint_pos_min >= pos_min_thold && checkpoint_pos_min != 0) {
        return -1;
    }

    const llama_pos checkpoint_pos_next = std::min(
        pos_next, std::max(checkpoint_pos_min + 1, checkpoint_pos_max));
    return (int) std::min(tokens.size_up_to_pos(checkpoint_pos_next), (size_t) n_tokens);
}

bool server_prompt_cache::load(
        server_prompt & prompt,
        const server_tokens & tokens_new,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        int32_t id_slot,
        bool cache_prompt,
        bool needs_checkpoint,
        int32_t n_swa,
        float slot_prompt_similarity) {
    if (tokens_new.empty()) {
        return true;
    }

    int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    const bool has_disk_cache = !cache_dir.empty();
    int effective_prefix_reuse_base = 0;
    if (has_disk_cache && !prompt.tokens.empty() && lcp_best > 0) {
        effective_prefix_reuse_base = lcp_best;

        const llama_pos pos_next = prompt.tokens.pos_next(lcp_best);
        const bool has_new_tokens = lcp_best < (int) tokens_new.size();
        const llama_pos pos_min_thold = server_prompt_pos_min_thold(pos_next, n_swa, has_new_tokens);
        const llama_pos pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), id_slot);

        if (pos_min >= pos_min_thold) {
            effective_prefix_reuse_base = 0;
            for (auto it = prompt.checkpoints.rbegin(); it != prompt.checkpoints.rend(); ++it) {
                const int reuse = server_prompt_checkpoint_reuse(
                    prompt.tokens, (*it)->n_tokens, (*it)->pos_min, (*it)->pos_max, pos_next, pos_min_thold);
                if (reuse >= 0) {
                    effective_prefix_reuse_base = reuse;
                    break;
                }
            }
        }

        if (effective_prefix_reuse_base == (int) tokens_new.size()) {
            effective_prefix_reuse_base--;
        }
    }

    if (has_disk_cache) {
        SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f, effective_prefix_reuse = %d\n",
                f_keep_best, f_sim_best, effective_prefix_reuse_base);
    } else {
        SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);
    }

    auto it_best = states.end();
    int effective_prefix_reuse_best = -1;
    size_t mapping_size_best = 0;

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        if (it->data.is_disk()) {
            if (!cache_prompt) {
                continue;
            }

            int effective_prefix_reuse = lcp_cur;
            if (needs_checkpoint) {
                effective_prefix_reuse = 0;

                const llama_pos pos_next = it->prompt.tokens.pos_next(lcp_cur);
                const bool has_new_tokens = lcp_cur < (int) tokens_new.size();
                const llama_pos pos_min_thold = server_prompt_pos_min_thold(pos_next, n_swa, has_new_tokens);
                for (auto checkpoint = it->checkpoints_disk.rbegin(); checkpoint != it->checkpoints_disk.rend(); ++checkpoint) {
                    const int reuse = server_prompt_checkpoint_reuse(
                        it->prompt.tokens,
                        checkpoint->n_tokens,
                        checkpoint->pos_min,
                        checkpoint->pos_max,
                        pos_next,
                        pos_min_thold);
                    if (reuse >= 0) {
                        effective_prefix_reuse = reuse;
                        break;
                    }
                }
            }

            if (effective_prefix_reuse == (int) tokens_new.size()) {
                effective_prefix_reuse--;
            }

            SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f, effective_prefix_reuse = %d\n",
                    it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur, effective_prefix_reuse);

            const float effective_similarity = float(effective_prefix_reuse) / tokens_new.size();
            const bool is_admissible =
                effective_prefix_reuse > effective_prefix_reuse_base &&
                effective_similarity > slot_prompt_similarity;
            const bool is_better = is_admissible &&
                (it_best == states.end() ||
                 effective_prefix_reuse > effective_prefix_reuse_best ||
                 (effective_prefix_reuse == effective_prefix_reuse_best && it->data.mapping_size <= mapping_size_best));
            if (is_better) {
                effective_prefix_reuse_best = effective_prefix_reuse;
                mapping_size_best = it->data.mapping_size;
                f_keep_best = f_keep_cur;
                f_sim_best  = f_sim_cur;
                it_best = it;
            }
            continue;
        }

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n",
                it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            lcp_best    = lcp_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        if (it_best->data.is_disk()) {
            SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f, effective_prefix_reuse = %d\n",
                    f_keep_best, f_sim_best, effective_prefix_reuse_best);
        } else {
            SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);
        }

        if (it_best->data.is_disk()) {
            auto & data = it_best->data;

            const size_t n_tgt = server_prompt_state_set_staged(ctx_tgt, data.mapping, data.main_size, id_slot);
            if (n_tgt != data.main_size) {
                // often only a full pool: the entry stays until room has been made for it
                SRV_ERR("failed to restore state with size %zu\n", data.main_size);
                id_failed = it_best->id;
                return false;
            }

            if (data.drft_size > 0) {
                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "cannot restore draft state without a draft context\n");
                    data.discard();
                    states.erase(it_best);
                    return false;
                }

                const size_t n_dft = server_prompt_state_set_staged(ctx_dft, data.mapping + data.main_size, data.drft_size, id_slot);
                if (n_dft != data.drft_size) {
                    SRV_WRN("failed to restore state with size %zu\n", data.drft_size);
                    id_failed = it_best->id;
                    return false;
                }
            }

            server_prompt restored;
            restored.tokens = it_best->prompt.tokens.clone();
            try {
                for (const auto & checkpoint_disk : it_best->checkpoints_disk) {
                    auto ckpt = std::make_shared<common_prompt_checkpoint>();
                    common_prompt_checkpoint & checkpoint = *ckpt;
                    checkpoint.n_tokens = checkpoint_disk.n_tokens;
                    checkpoint.id_task  = checkpoint_disk.id_task;
                    checkpoint.pos_min  = checkpoint_disk.pos_min;
                    checkpoint.pos_max  = checkpoint_disk.pos_max;

                    checkpoint.data_tgt.assign(
                        data.mapping + checkpoint_disk.offset_tgt,
                        data.mapping + checkpoint_disk.offset_tgt + checkpoint_disk.size_tgt);
                    checkpoint.data_dft.assign(
                        data.mapping + checkpoint_disk.offset_dft,
                        data.mapping + checkpoint_disk.offset_dft + checkpoint_disk.size_dft);
                    checkpoint.data_spec.assign(
                        data.mapping + checkpoint_disk.offset_spec,
                        data.mapping + checkpoint_disk.offset_spec + checkpoint_disk.size_spec);

                    if (!ckpt_dir.empty() && !checkpoint.spill(ckpt_dir)) {
                        SRV_WRN("could not move a context checkpoint to %s - it stays in memory\n", ckpt_dir.c_str());
                    }

                    restored.checkpoints.push_back(std::move(ckpt));
                }
            } catch (const std::bad_alloc & e) {
                SRV_ERR("failed to allocate memory for restored prompt checkpoints: %s\n", e.what());
                return false;
            }

            prompt = std::move(restored);

            std::error_code ec;
            std::filesystem::last_write_time(
                data.cache_file + ".meta", std::filesystem::file_time_type::clock::now(), ec);
            if (ec) {
                SRV_WRN("failed to update disk cache access time %s: %s\n", data.cache_file.c_str(), ec.message().c_str());
            }

            // Disk entries are reusable and persistent. Move the hit to the
            // back so size-based eviction remains least-recently-used.
            states.splice(states.end(), states, it_best);
            return true;
        } else {
            {
                auto & data = it_best->data.main;

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_ERR("failed to restore state with size %zu\n", size);

                    id_failed = it_best->id;

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }

            {
                auto & data = it_best->data.drft;

                if (!data.empty()) {
                    GGML_ASSERT(ctx_dft);

                    const size_t size = data.size();
                    const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                    if (n != size) {
                        SRV_WRN("failed to restore state with size %zu\n", size);

                        id_failed = it_best->id;

                        return false;
                    }

                    data.clear();
                    data.shrink_to_fit();
                }
            }
        }

        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_RESTORE__\n");

        // the whole cached prompt is a prefix of the new one
        if (lcp_best == (int) it_best->prompt.tokens.size()) {
            SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_FULL_RESTORE__\n");
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);
    }

    return true;
}

bool server_prompt_cache::contains(uint64_t id) const {
    return std::any_of(states.begin(), states.end(), [id](const server_prompt_cache_state & state) {
        return state.id == id;
    });
}

void server_prompt_cache::discard_unrestorable(uint64_t id) {
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->id == id) {
            if (it->data.is_disk()) {
                SRV_WRN("removing a disk prompt cache entry that cannot be restored (%s)\n", it->data.cache_file.c_str());
                it->data.discard();
                states.erase(it);
            }
            return;
        }
    }
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.front().data.discard();
            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.front().data.discard();
            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(),
                state.data.is_disk() ? state.checkpoints_disk.size() : state.prompt.checkpoints.size(),
                state.size() / (1024.0 * 1024.0));
    }
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task) const {
    const server_slot_info * ret = nullptr;

    bool update_cache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task
            if (ret->is_processing) {
                return { ret->id, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
        }
    }

    // the slot this conversation used last, if it is idle and still holds it
    bool by_conversation = false;
    if (task.id_slot == -1 && !task.conversation_id.empty()) {
        for (const auto & slot : slots) {
            if (slot.is_processing || slot.prompt->tokens.empty() || slot.conversation_id == nullptr ||
                *slot.conversation_id != task.conversation_id) {
                continue;
            }
            const size_t lcp_len = slot.prompt->tokens.get_common_prefix(task.tokens);
            const float  f_keep  = float(lcp_len) / slot.prompt->tokens.size();
            SLT_INF(slot, "selected slot by conversation id, f_keep = %.3f\n", f_keep);
            ret             = &slot;
            by_conversation = true;
            update_cache    = f_keep < 0.5f;
            break;
        }
    }

    // a prefix shared by many conversations (slot_match_skip tokens: a system prompt, tools) is no evidence that
    // a slot holds this one, so the similarity counts the tokens after it
    const size_t n_skip = std::min<size_t>(std::max<int32_t>(task.slot_match_skip, 0), task.tokens.size());

    // find the slot that has at least n% prompt similarity
    if (slot_prompt_similarity != 0.0f && !by_conversation) {
        float  f_sim_best = 0;
        size_t lcp_best   = 0;

        for (const auto & slot : slots) {
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }

            // skip the slot if it is not available
            if (slot.is_processing) {
                SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing);
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // fraction of the Longest Common Prefix length with respect to the input prompt length, after the shared prefix
            const size_t lcp_len = tokens.get_common_prefix(task.tokens);
            const size_t n_new   = task.tokens.size() - n_skip;
            const float f_sim_cur = n_new > 0 ? float(lcp_len > n_skip ? lcp_len - n_skip : 0) / n_new : 0.0f;

            SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

            // select the current slot if the criteria match
            if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                f_sim_best = f_sim_cur;
                lcp_best   = lcp_len;

                ret = &slot;
            }
        }

        if (ret != nullptr) {
            const float f_keep = ret->prompt->tokens.empty() ? 0.0f : float(lcp_best) / ret->prompt->tokens.size();

            if (task.id_slot == -1) {
                SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        f_sim_best, slot_prompt_similarity, f_keep);
            }

            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            if (f_keep < 0.5f) {
                update_cache = true;
            }

            // the best match would discard most of its slot: leave that conversation in place if an empty slot is idle
            if (task.id_slot == -1 && f_keep < slot_keep_min) {
                for (const auto & slot : slots) {
                    if (!slot.is_processing && slot.prompt->tokens.empty()) {
                        SLT_INF(slot, "selected empty slot instead: slot %d would keep %.3f of its tokens (< %.3f)\n",
                                ret->id, f_keep, slot_keep_min);
                        ret          = &slot;
                        update_cache = true; // a prompt cache entry for this prompt can still be restored
                        break;
                    }
                }
            }
        }
    }

    // find an empty slot, or the one that has been least recently used
    if (ret == nullptr) {
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt->n_tokens(), ret->t_last_used);

            update_cache = true;
        }
    }

    if (ret == nullptr) {
        return {};
    }

    // cache prompts only for the tasks that reuse them
    update_cache = update_cache && !task.is_stateless();

    return { ret->id, update_cache };
}
