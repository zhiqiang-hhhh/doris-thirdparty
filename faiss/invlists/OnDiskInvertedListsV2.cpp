/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/OnDiskInvertedListsV2.h>

#include <cstring>

#include <faiss/impl/FaissAssert.h>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace faiss {

/*******************************************************
 * FileRandomAccessReader — default POSIX pread backend
 *******************************************************/

FileRandomAccessReader::FileRandomAccessReader(const std::string& filename) {
#ifndef _WIN32
    fd_ = ::open(filename.c_str(), O_RDONLY);
    FAISS_THROW_IF_NOT_FMT(
            fd_ >= 0,
            "FileRandomAccessReader: cannot open %s: %s",
            filename.c_str(),
            strerror(errno));
#else
    FAISS_THROW_MSG(
            "FileRandomAccessReader is not supported on Windows");
#endif
}

FileRandomAccessReader::~FileRandomAccessReader() {
#ifndef _WIN32
    if (fd_ >= 0) {
        ::close(fd_);
    }
#endif
}

void FileRandomAccessReader::read_at(
        size_t offset,
        void* ptr,
        size_t nbytes) const {
#ifndef _WIN32
    size_t done = 0;
    auto* out = static_cast<uint8_t*>(ptr);
    while (done < nbytes) {
        ssize_t nr = ::pread(fd_, out + done, nbytes - done, offset + done);
        FAISS_THROW_IF_NOT_MSG(nr >= 0, "pread failed in FileRandomAccessReader");
        FAISS_THROW_IF_NOT_MSG(nr > 0, "unexpected EOF in FileRandomAccessReader");
        done += static_cast<size_t>(nr);
    }
#else
    (void)offset;
    (void)ptr;
    (void)nbytes;
    FAISS_THROW_MSG("FileRandomAccessReader is not supported on Windows");
#endif
}

/*******************************************************
 * OnDiskInvertedListsV2Iterator
 *******************************************************/

namespace {

struct OnDiskInvertedListsV2Iterator : InvertedListsIterator {
    const OnDiskInvertedListsV2* parent;
    size_t list_size = 0;
    size_t i = 0;
    uint8_t* codes = nullptr;
    idx_t* ids = nullptr;
    size_t codes_bytes = 0;
    size_t ids_bytes = 0;

    OnDiskInvertedListsV2Iterator(
            const OnDiskInvertedListsV2* parent,
            size_t list_no)
            : parent(parent) {
        const OnDiskOneList& l = parent->lists.at(list_no);
        if (l.size == 0) {
            return;
        }
        list_size = l.size;
        codes_bytes = l.size * parent->code_size;
        ids_bytes = l.size * sizeof(idx_t);

        codes = static_cast<uint8_t*>(parent->alloc_buf(codes_bytes));
        ids = static_cast<idx_t*>(parent->alloc_buf(ids_bytes));

        // Layout: [codes: capacity*code_size][ids: capacity*sizeof(idx_t)]
        parent->read_at_exact(l.offset, codes, codes_bytes);
        parent->read_at_exact(
                l.offset + l.capacity * parent->code_size,
                ids,
                ids_bytes);
    }

    ~OnDiskInvertedListsV2Iterator() override {
        if (codes) {
            parent->free_buf(codes, codes_bytes);
        }
        if (ids) {
            parent->free_buf(ids, ids_bytes);
        }
    }

    // non-copyable
    OnDiskInvertedListsV2Iterator(const OnDiskInvertedListsV2Iterator&) = delete;
    OnDiskInvertedListsV2Iterator& operator=(const OnDiskInvertedListsV2Iterator&) = delete;

    bool is_available() const override {
        return i < list_size;
    }

    void next() override {
        i++;
    }

    std::pair<idx_t, const uint8_t*> get_id_and_codes() override {
        return {ids[i], codes + i * parent->code_size};
    }
};

} // namespace

/*******************************************************
 * OnDiskInvertedListsV2
 *******************************************************/

OnDiskInvertedListsV2::OnDiskInvertedListsV2(const OnDiskInvertedLists& src)
        : ReadOnlyInvertedLists(src.nlist, src.code_size),
          lists(src.lists) {
    use_iterator = true;
}

OnDiskInvertedListsV2::OnDiskInvertedListsV2(
        size_t nlist,
        size_t code_size,
        std::vector<OnDiskOneList> lists)
        : ReadOnlyInvertedLists(nlist, code_size),
          lists(std::move(lists)) {
    use_iterator = true;
}

void OnDiskInvertedListsV2::set_reader(
        std::unique_ptr<RandomAccessReader> reader) {
    reader_ = std::move(reader);
}

void OnDiskInvertedListsV2::set_allocator(MemoryAllocator allocator) {
    allocator_ = std::move(allocator);
}

void* OnDiskInvertedListsV2::alloc_buf(size_t nbytes) const {
    if (allocator_.alloc) {
        return allocator_.alloc(nbytes);
    }
    return new uint8_t[nbytes];
}

void OnDiskInvertedListsV2::free_buf(void* buf, size_t nbytes) const {
    if (allocator_.free) {
        allocator_.free(buf, nbytes);
        return;
    }
    delete[] static_cast<uint8_t*>(buf);
}

size_t OnDiskInvertedListsV2::list_size(size_t list_no) const {
    return lists.at(list_no).size;
}

const uint8_t* OnDiskInvertedListsV2::get_codes(size_t list_no) const {
    const OnDiskOneList& l = lists.at(list_no);
    if (l.size == 0) {
        return nullptr;
    }
    size_t nbytes = l.size * code_size;
    auto* out = static_cast<uint8_t*>(alloc_buf(nbytes));
    read_at_exact(l.offset, out, nbytes);
    return out;
}

const idx_t* OnDiskInvertedListsV2::get_ids(size_t list_no) const {
    const OnDiskOneList& l = lists.at(list_no);
    if (l.size == 0) {
        return nullptr;
    }
    size_t nbytes = l.size * sizeof(idx_t);
    auto* out = static_cast<idx_t*>(alloc_buf(nbytes));
    read_at_exact(
            l.offset + l.capacity * code_size,
            out,
            nbytes);
    return out;
}

void OnDiskInvertedListsV2::release_codes(
        size_t list_no,
        const uint8_t* codes) const {
    if (!codes) return;
    const OnDiskOneList& l = lists.at(list_no);
    free_buf(const_cast<uint8_t*>(codes), l.size * code_size);
}

void OnDiskInvertedListsV2::release_ids(
        size_t list_no,
        const idx_t* ids) const {
    if (!ids) return;
    const OnDiskOneList& l = lists.at(list_no);
    free_buf(const_cast<idx_t*>(ids), l.size * sizeof(idx_t));
}

bool OnDiskInvertedListsV2::is_empty(size_t list_no, void*) const {
    return lists.at(list_no).size == 0;
}

InvertedListsIterator* OnDiskInvertedListsV2::get_iterator(
        size_t list_no,
        void*) const {
    return new OnDiskInvertedListsV2Iterator(this, list_no);
}

void OnDiskInvertedListsV2::prefetch_lists(const idx_t*, int) const {
    // no-op: random-access reader handles caching
}

void OnDiskInvertedListsV2::read_at_exact(
        size_t offset,
        void* ptr,
        size_t nbytes) const {
    if (nbytes == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            reader_,
            "OnDiskInvertedListsV2: reader not set, call set_reader() first");
    reader_->read_at(offset, ptr, nbytes);
}

/*******************************************************
 * Helper functions
 *******************************************************/

OnDiskInvertedListsV2* replace_ondisk_invlists_with_v2(Index* index) {
    auto* ivf = dynamic_cast<IndexIVF*>(index);
    FAISS_THROW_IF_NOT_MSG(
            ivf, "replace_ondisk_invlists_with_v2 expects IndexIVF");

    auto* od = dynamic_cast<OnDiskInvertedLists*>(ivf->invlists);
    FAISS_THROW_IF_NOT_MSG(
            od,
            "replace_ondisk_invlists_with_v2 expects IndexIVF with "
            "OnDiskInvertedLists");

    auto* v2 = new OnDiskInvertedListsV2(*od);
    ivf->replace_invlists(v2, true);
    return v2;
}

} // namespace faiss
