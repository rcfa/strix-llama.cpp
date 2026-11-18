#pragma once

#include "server-common.h"
#include "server-task.h"

#include <list>
#include <memory>
#include <string>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    std::shared_ptr<const common_prompt_checkpoint> ckpt;
};

struct server_prompt {
    server_tokens tokens;

    // a checkpoint does not change after it is created, so the prompt cache shares it with the slot instead of a copy
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const;
};

llama_pos server_prompt_pos_min_thold(
        llama_pos pos_next,
        int32_t   n_swa,
        bool      has_new_tokens);

// Returns -1 when the checkpoint cannot be used, otherwise the resulting n_past.
int server_prompt_checkpoint_reuse(
        const server_tokens & tokens,
        int64_t               n_tokens,
        llama_pos             checkpoint_pos_min,
        llama_pos             checkpoint_pos_max,
        llama_pos             pos_next,
        llama_pos             pos_min_thold);

// Copy a sequence's state between a context and a disk cache mapping through
// ordinary heap memory. Backends must not read or write the file mapping
// directly: the ROCm runtime pins the mapped pages for the copy, and writeback
// then keeps rewriting the file for as long as the server runs. Returns the
// number of bytes copied, 0 on failure.
size_t server_prompt_state_get_staged(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id);
size_t server_prompt_state_set_staged(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id);

struct server_prompt_data {
    common_state_data main;
    common_state_data drft;

    std::string cache_file;
    uint8_t * mapping = nullptr;
    size_t mapping_size = 0;
    size_t main_size = 0;
    size_t drft_size = 0;

    // Newly-created files are removed if saving is interrupted. Once the
    // metadata sidecar is committed, they survive object destruction and are
    // removed only by explicit cache eviction.
    bool remove_file_on_destroy = true;

#ifdef _WIN32
    void * file_handle = nullptr;
    void * mapping_handle = nullptr;
#endif

    server_prompt_data() = default;
    ~server_prompt_data();

    server_prompt_data(const server_prompt_data &) = delete;
    server_prompt_data & operator=(const server_prompt_data &) = delete;

    server_prompt_data(server_prompt_data && other) noexcept;
    server_prompt_data & operator=(server_prompt_data && other) noexcept;

    bool is_disk() const {
        return !cache_file.empty();
    }

    void clear();
    void discard();

    size_t size() const {
        if (is_disk()) {
            return mapping_size;
        }
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_checkpoint {
    int64_t n_tokens = 0;
    int id_task = -1;

    llama_pos pos_min = 0;
    llama_pos pos_max = 0;

    size_t offset_tgt = 0;
    size_t size_tgt = 0;
    size_t offset_dft = 0;
    size_t size_dft = 0;
    size_t offset_spec = 0;
    size_t size_spec = 0;
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;
    std::vector<server_prompt_cache_checkpoint> checkpoints_disk;

    // unique within the cache, set by alloc()
    uint64_t id = 0;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(
        int32_t limit_size_mib,
        size_t limit_tokens,
        const std::string & cache_dir = "",
        const std::string & cache_key = "",
        bool has_mtmd = false);

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    std::string cache_dir;
    std::string cache_key;

    // context checkpoints restored from the cache are moved to unlinked files here (--ctx-checkpoints-dir)
    std::string ckpt_dir;

    bool has_mtmd = false;

    uint64_t next_file_id = 0;

    uint64_t id_last = 0;

    // the entry the last failed load() could not restore
    uint64_t id_failed = 0;

    bool contains(uint64_t id) const;

    // removes a disk entry that cannot be restored even with room made for it
    void discard_unrestorable(uint64_t id);

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool commit(server_prompt_cache_state * state);
    void discard(server_prompt_cache_state * state);

    bool load(
            server_prompt & prompt,
            const server_tokens & tokens_new,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            int32_t id_slot,
            bool cache_prompt,
            bool needs_checkpoint,
            int32_t n_swa,
            float slot_prompt_similarity);

    void update();
};

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;

    // the conversation_id of the request that left the slot's tokens, empty if unknown
    const std::string * conversation_id = nullptr;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // the slot chosen by similarity must keep this fraction of its tokens, else an idle empty slot is used
    float slot_keep_min = 0.0f;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task) const;
};
