#ifndef CETO_DETAIL_SP_COUNTED_BASE_NT_HPP_INCLUDED
#define CETO_DETAIL_SP_COUNTED_BASE_NT_HPP_INCLUDED

//
//  detail/sp_counted_base_nt.hpp
//
//  Copyright (c) 2001, 2002, 2003 Peter Dimov and Multi Media Ltd.
//  Copyright 2004-2005 Peter Dimov
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

namespace ceto {

  namespace detail {

    class sp_counted_base {
    private:
      sp_counted_base(sp_counted_base const &);
      sp_counted_base &operator=(sp_counted_base const &);

      long use_count_; // #shared
      long weak_count_; // #weak + (#shared != 0)

    public:
      sp_counted_base()
          : use_count_(1)
          , weak_count_(1) {
      }

      virtual ~sp_counted_base() // nothrow
      {
      }

      // dispose() is called when use_count_ drops to zero, to release
      // the resources managed by *this.

      virtual void dispose() = 0; // nothrow

      // destroy() is called when weak_count_ drops to zero.

      virtual void destroy() // nothrow
      {
        delete this;
      }

      virtual void *get_deleter(std::type_info const &ti) = 0;
      virtual void *get_untyped_deleter() = 0;

      void add_ref_copy() {
        ++use_count_;
      }

      bool add_ref_lock() // true on success
      {
        if (use_count_ == 0)
          return false;
        ++use_count_;
        return true;
      }

      void release() // nothrow
      {
        if (--use_count_ == 0) {
          dispose();
          weak_release();
        }
      }

      void weak_add_ref() // nothrow
      {
        ++weak_count_;
      }

      void weak_release() // nothrow
      {
        if (--weak_count_ == 0) {
          destroy();
        }
      }

      long use_count() const // nothrow
      {
        return use_count_;
      }
    };

  } // namespace detail

} // namespace ceto

#endif // #ifndef CETO_DETAIL_SP_COUNTED_BASE_NT_HPP_INCLUDED
