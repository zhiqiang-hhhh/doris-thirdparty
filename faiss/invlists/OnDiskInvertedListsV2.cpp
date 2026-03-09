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
    size_t list_no;
    size_t i = 0;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;

    OnDiskInvertedListsV2Iterator(
            const OnDiskInvertedListsV2* parent,
            size_t list_no)
            : parent(parent), list_no(list_no) {
        const OnDiskOneList& l = parent->lists.at(list_no);
        if (l.size == 0) {
            return;
        }

        codes.resize(l.size * parent->code_size);
        ids.resize(l.size);

        // Layout: [codes: capacity*code_size][ids: capacity*sizeof(idx_t)]
        parent->read_at_exact(l.offset, codes.data(), codes.size());
        parent->read_at_exact(
                l.offset + l.capacity * parent->code_size,
                ids.data(),
                ids.size() * sizeof(idx_t));
    }

    bool is_available() const override {
        return i < ids.size();
    }

    void next() override {
        i++;
    }

    std::pair<idx_t, const uint8_t*> get_id_and_codes() override {
        return {ids[i], codes.data() + i * parent->code_size};
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

size_t OnDiskInvertedListsV2::list_size(size_t list_no) const {
    return lists.at(list_no).size;
}

const uint8_t* OnDiskInvertedListsV2::get_codes(size_t list_no) const {
    const OnDiskOneList& l = lists.at(list_no);
    if (l.size == 0) {
        return nullptr;
    }
    auto* out = new uint8_t[l.size * code_size];
    read_at_exact(l.offset, out, l.size * code_size);
    return out;
}

const idx_t* OnDiskInvertedListsV2::get_ids(size_t list_no) const {
    const OnDiskOneList& l = lists.at(list_no);
    if (l.size == 0) {
        return nullptr;
    }
    auto* out = new idx_t[l.size];
    read_at_exact(
            l.offset + l.capacity * code_size,
            out,
            l.size * sizeof(idx_t));
    return out;
}

void OnDiskInvertedListsV2::release_codes(
        size_t /*list_no*/,
        const uint8_t* codes) const {
    delete[] codes;
}

void OnDiskInvertedListsV2::release_ids(
        size_t /*list_no*/,
        const idx_t* ids) const {
    delete[] ids;
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
