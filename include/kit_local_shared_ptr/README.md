# local_shared_ptr

- local: faster than shared_ptr when no thread safety is needed
- local_weak_ptr (boost currently has no local_weak_ptr)
- enable_shared_from_this
- no local_shared_array support
- only supports the latest clang/gcc/vc
- naming matches boost::local_shared_ptr if they ever add these features

Forked from https://github.com/yohei-yoshihara/minimal_boost_shared_ptr

Taken from https://github.com/flipcoder/local_shared_ptr (with namespace names changed from kit to ceto)

last commit used:
https://github.com/flipcoder/local_shared_ptr/commit/fdaef36ff4edecd5c055674fd2bf2e53a6433eed
