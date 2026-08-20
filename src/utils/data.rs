use rustc_hash::{FxHashMap, FxHashSet};

/// Fast HashMap using FxHash - optimized for integer keys like JobId, WorkerId, FlowId.
/// FxHash is much faster than xxhash for small integer keys.
pub type DHashMap<K, V> = FxHashMap<K, V>;

/// Fast HashSet using FxHash.
pub type DHashSet<T> = FxHashSet<T>;
