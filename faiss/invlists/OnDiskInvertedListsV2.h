/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/OnDiskInvertedLists.h>

namespace faiss {

/**
 * Abstract interface for random-access reads from a storage backend.
 *
 * Implementations may back this with pread(fd), CLucene IndexInput,
 * or any other seekable byte source.
 */
struct RandomAccessReader {
    virtual ~RandomAccessReader() = default;

    /**
     * Read exactly @p nbytes starting at byte @p offset into @p ptr.
     * Must throw on short read or I/O error.
     */
    virtual void read_at(size_t offset, void* ptr, size_t nbytes) const = 0;
};

/**
 * Default RandomAccessReader backed by pread(fd) on a local file.
 * Only available on POSIX systems.
 */
struct FileRandomAccessReader : RandomAccessReader {
    explicit FileRandomAccessReader(const std::string& filename);
    ~FileRandomAccessReader() override;

    void read_at(size_t offset, void* ptr, size_t nbytes) const override;

private:
    int fd_ = -1;
};

/**
 * OnDiskInvertedListsV2:
 * - no mmap
 * - read-once list scan via iterator
 * - backed by a pluggable RandomAccessReader (file, CLucene IndexInput, etc.)
 *
 * Layout per list in the data file (same as OnDiskInvertedLists):
 *   [codes: capacity * code_size bytes] [ids: capacity * sizeof(idx_t) bytes]
 * Only the first `size` entries in each region are valid.
 */
struct OnDiskInvertedListsV2 : ReadOnlyInvertedLists {
    // same metadata as OnDiskInvertedLists
    std::vector<OnDiskOneList> lists;

    explicit OnDiskInvertedListsV2(const OnDiskInvertedLists& src);

    OnDiskInvertedListsV2(
            size_t nlist,
            size_t code_size,
            std::vector<OnDiskOneList> lists);

    /// Bind the reader that provides random access to the ivfdata content.
    /// Must be called before any read operation (get_codes, get_ids, get_iterator).
    void set_reader(std::unique_ptr<RandomAccessReader> reader);

    size_t list_size(size_t list_no) const override;

    // fallback APIs (allocate per call, caller releases with release_*)
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;
    void release_codes(size_t list_no, const uint8_t* codes) const override;
    void release_ids(size_t list_no, const idx_t* ids) const override;

    bool is_empty(size_t list_no, void* inverted_list_context = nullptr)
            const override;
    InvertedListsIterator* get_iterator(
            size_t list_no,
            void* inverted_list_context = nullptr) const override;

    void prefetch_lists(const idx_t* list_nos, int nlist) const override;

    /// Read exactly @p nbytes from the data at byte @p offset.
    void read_at_exact(size_t offset, void* ptr, size_t nbytes) const;

private:
    std::unique_ptr<RandomAccessReader> reader_;
};

/**
 * Replace an IndexIVF's OnDiskInvertedLists with OnDiskInvertedListsV2.
 * The V2 object is created but has no reader yet — caller must call
 * set_reader() on the resulting inverted lists before any search.
 */
FAISS_API OnDiskInvertedListsV2* replace_ondisk_invlists_with_v2(Index* index);

} // namespace faiss
