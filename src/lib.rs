//! `iprange` is a library for managing IP ranges.
//!
//! An [`IpRange`] is a set of networks.
//! The type of the networks it holds is specified by the generics type of [`IpRange`].
//!
//! You can add or remove an [`IpNet`] from an [`IpRange`].
//! An [`IpNet`] can be either an `Ipv4Net` or an `Ipv6Net`.
//!
//! It also supports these useful operations:
//!
//! * [`merge`]
//! * [`intersect`]
//! * [`exclude`]
//!
//! Here is a simple example:
//!
//! ```
//! # extern crate ipnet;
//! use std::net::Ipv4Addr;
//! use iprange::IpRange;
//! use ipnet::Ipv4Net;
//!
//! let ip_range: IpRange<Ipv4Net> = ["10.0.0.0/8", "172.16.0.0/16", "192.168.1.0/24"]
//!     .iter()
//!     .map(|s| s.parse().unwrap())
//!     .collect();
//!
//! assert!(ip_range.contains(&"172.16.32.1".parse::<Ipv4Addr>().unwrap()));
//! assert!(ip_range.contains(&"192.168.1.1".parse::<Ipv4Addr>().unwrap()));
//! ```
//!
//! [`IpRange`]: struct.IpRange.html
//! [`IpNet`]: trait.IpNet.html
//! [`Ipv4Net`]: https://docs.rs/ipnet/1.0.0/ipnet/struct.Ipv4Net.html
//! [`merge`]: struct.IpRange.html#method.merge
//! [`intersect`]: struct.IpRange.html#method.intersect
//! [`exclude`]: struct.IpRange.html#method.exclude

#[cfg(feature = "ipnet")]
extern crate ipnet;
#[cfg(feature = "ipnetwork")]
extern crate ipnetwork;
#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

use std::collections::VecDeque;
use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;

#[cfg(feature = "ipnet")]
#[path = "ipnet.rs"]
mod ipnet_impl;
#[cfg(feature = "ipnetwork")]
#[path = "ipnetwork.rs"]
mod ipnetwork_impl;

/// A set of networks that supports various operations:
///
/// * [`add`]
/// * [`remove`]
/// * [`contains`]
/// * [`merge`]
/// * [`intersect`]
/// * [`exclude`]
///
/// `IntoIter` is implemented for `&IpRange`. So, you can use `for`
/// to iterate over the networks in an `IpRange`:
///
/// ```
/// # extern crate ipnet;
/// use iprange::IpRange;
/// use ipnet::Ipv4Net;
///
/// let ip_range: IpRange<Ipv4Net> = ["172.16.0.0/16", "192.168.1.0/24"]
///     .iter()
///     .map(|s| s.parse().unwrap())
///     .collect();
///
/// for network in &ip_range {
///     println!("{:?}", network);
/// }
/// ```
///
/// [`add`]: struct.IpRange.html#method.add
/// [`remove`]: struct.IpRange.html#method.remove
/// [`contains`]: struct.IpRange.html#method.contains
/// [`merge`]: struct.IpRange.html#method.merge
/// [`intersect`]: struct.IpRange.html#method.intersect
/// [`exclude`]: struct.IpRange.html#method.exclude
#[derive(Clone, PartialEq, Eq)]
pub struct IpRange<N: IpNet> {
    // IpRange uses a radix trie to store networks
    trie: IpTrie<N>,
    phantom_net: PhantomData<N>,
}

impl<N: IpNet> IpRange<N> {
    /// Creates an empty `IpRange`.
    pub fn new() -> IpRange<N> {
        IpRange {
            trie: IpTrie::new(),
            phantom_net: PhantomData,
        }
    }

    /// Add a network to `self`.
    ///
    /// Returns `&mut self` in order to enable method chaining.
    ///
    /// Pay attention that this operation will not combine two
    /// networks automatically. To do this, call [`simplify`] method
    /// explicitly. For example:
    ///
    /// ```
    /// # extern crate ipnet;
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    /// ip_range.add("192.168.0.0/24".parse().unwrap())
    ///        .add("192.168.1.0/24".parse().unwrap());
    /// assert_eq!(ip_range.into_iter().count(), 2);
    ///
    /// ip_range.simplify();
    /// assert_eq!(ip_range.into_iter().count(), 1);
    /// ```
    ///
    /// [`simplify`]: struct.IpRange.html#method.simplify
    pub fn add(&mut self, network: N) -> &mut Self {
        self.trie.insert(network);
        self
    }

    /// Remove a network from `self`.
    ///
    /// Returns `&mut self` in order to enable method chaining.
    ///
    /// `self` does not necessarily has exactly the network to be removed.
    /// The network can be a networkwork of a network in `self`.
    /// This method will do splitting and remove the corresponding network.
    /// For example:
    ///
    /// ```
    /// # extern crate ipnet;
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    /// ip_range.add("192.168.0.0/23".parse().unwrap())
    ///         .remove("192.168.0.0/24".parse().unwrap());
    /// // Now, ip_range has only one network: "192.168.1.0/24".
    /// ```
    pub fn remove(&mut self, network: N) -> &mut Self {
        self.trie.remove(network);
        self
    }

    /// Returns `true` if the `self` has no network.
    ///
    /// # Examples
    /// ```
    /// # extern crate ipnet;
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// let mut ip_range = IpRange::new();
    /// let network: Ipv4Net = "1.0.1.0/24".parse().unwrap();
    /// ip_range.add(network.clone());
    /// ip_range.remove(network);
    /// assert!(ip_range.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.trie.root.is_none()
    }

    /// Simplify `self` by combining networks. For example:
    ///
    /// ```
    /// # extern crate ipnet;
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    /// ip_range
    ///     .add("192.168.0.0/20".parse().unwrap())
    ///     .add("192.168.16.0/22".parse().unwrap())
    ///     .add("192.168.20.0/24".parse().unwrap())
    ///     .add("192.168.21.0/24".parse().unwrap())
    ///     .add("192.168.22.0/24".parse().unwrap())
    ///     .add("192.168.23.0/24".parse().unwrap())
    ///     .add("192.168.24.0/21".parse().unwrap())
    ///     .simplify();
    /// // Now, ip_range has only one network: "192.168.0.0/19".
    /// ```
    pub fn simplify(&mut self) {
        self.trie.simplify();
    }

    /// Returns a new `IpRange` which contains all networks
    /// that is either in `self` or in `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn merge(&self, other: &IpRange<N>) -> Self {
        self.into_iter().chain(other.into_iter()).collect()
    }

    /// Returns a new `IpRange` which contains all networks
    /// that is in both `self` and `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn intersect(&self, other: &IpRange<N>) -> Self {
        let range1 = self.into_iter().filter(|network| other.contains(network));
        let range2 = other.into_iter().filter(|network| self.contains(network));
        range1.chain(range2).collect()
    }

    /// Returns a new `IpRange` which contains all networks
    /// that is in `self` while not in `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn exclude(&self, other: &IpRange<N>) -> IpRange<N> {
        let mut new = (*self).clone();
        for network in other {
            new.remove(network);
        }
        new
    }

    /// Tests if `self` contains `network`.
    ///
    /// `network` is anything that can be converted into `N`.
    /// See `ToNetwork<N>` for detail.
    pub fn contains<T: ToNetwork<N>>(&self, network: &T) -> bool {
        self.supernet(&network.to_network()).is_some()
    }

    /// Returns the network in `self` which is the supernetwork of `network`.
    ///
    /// Returns None if no network in `self` contains `network`.
    pub fn supernet<T: ToNetwork<N>>(&self, network: &T) -> Option<N> {
        self.trie.search(network.to_network())
    }

    /// Returns the iterator to `&self`.
    pub fn iter(&self) -> IpRangeIter<N> {
        self.into_iter()
    }
}

impl<N> Default for IpRange<N>
where
    N: IpNet + ToNetwork<N> + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<N: IpNet> fmt::Debug for IpRange<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let mut networks: Vec<_> = self
            .iter()
            .take(4)
            .map(|net| format!("{:?}", net))
            .collect();
        if networks.len() == 4 {
            networks[3] = "...".to_string();
        }
        write!(f, "IpRange [{}]", networks.join(", "))
    }
}

impl<'a, N> IntoIterator for &'a IpRange<N>
where
    N: IpNet + ToNetwork<N> + Clone,
{
    type Item = N;
    type IntoIter = IpRangeIter<'a, N>;

    fn into_iter(self) -> Self::IntoIter {
        let mut queue = VecDeque::new();
        if let Some(root) = self.trie.root.as_ref() {
            let state: N::S = root.init_traverse_state();
            queue.push_back(state);
        }
        IpRangeIter {
            queue,
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "serde")]
impl<N: IpNet> serde::Serialize for IpRange<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde::Serialize::serialize(&self.trie.root, serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, N: IpNet> serde::Deserialize<'de> for IpRange<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(IpRange {
            trie: IpTrie {
                root: serde::Deserialize::deserialize(deserializer)?,
                phantom_net: PhantomData,
            },
            phantom_net: PhantomData,
        })
    }
}

/// An abstraction for IP networks.
pub trait IpNet: ToNetwork<Self> + fmt::Debug + Ord + Copy
where
    Self: Sized,
{
    /// Used for internal traversing.
    type S: TraverseState<Net = Self>;
    ///`I` is an iterator to the prefix bits of the network.
    type I: Iterator<Item = bool>;

    /// Returns the iterator to the prefix bits of the network.
    fn prefix_bits(&self) -> Self::I;

    /// Returns the prefix length.
    fn prefix_len(&self) -> u8;

    /// Returns a copy of the network with the address truncated to the given length.
    fn with_new_prefix(&self, len: u8) -> Self;
}

/// Anything that can be converted to `IpNet`.
///
/// Due to limitation of Rust's type system,
/// this trait is only implemented for some
/// concrete types.
pub trait ToNetwork<N: IpNet> {
    fn to_network(&self) -> N;
}

/// An iterator over the networks in an [`IpRange`].
///
/// BFS (Breadth-First-Search) is used for traversing the inner Radix Trie.
///
/// [`IpRange`]: struct.IpRange.html
pub struct IpRangeIter<'a, N>
where
    N: IpNet,
{
    queue: VecDeque<N::S>,
    _phantom: PhantomData<&'a N>,
}

/// Used for internal traversing.
///
/// You can simply ignore this trait.
#[doc(hidden)]
pub trait TraverseState {
    type Net: IpNet;

    fn node(&self) -> *const IpTrieNode;

    fn init(root: &IpTrieNode) -> Self;

    fn transit(&self, next_node: &IpTrieNode, current_bit: bool) -> Self;

    fn build(&self) -> Self::Net;
}

impl<'a, N> Iterator for IpRangeIter<'a, N>
where
    N: IpNet,
{
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(elem) = self.queue.pop_front() {
            // Get the front element of the queue.
            // If it is a leaf, it represents a network.
            // SAFETY: IpRangeIter has an PhantomData<'a N> so the IpNet must
            // exist when this iterator exists.
            let node = unsafe { &*elem.node() };
            if node.is_leaf() {
                return Some(elem.build());
            }
            for &i in &[0, 1] {
                if let Some(child) = node.children[i as usize].as_ref() {
                    // Push the child nodes into the queue
                    self.queue.push_back(elem.transit(child, i != 0));
                }
            }
        }
        None
    }
}

impl<N> FromIterator<N> for IpRange<N>
where
    N: IpNet + ToNetwork<N> + Clone,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = N>,
    {
        let mut ip_range = IpRange::new();
        for network in iter {
            ip_range.add(network);
        }
        ip_range.simplify();
        ip_range
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct IpTrie<N>
where
    N: IpNet,
{
    root: Option<IpTrieNode>,
    phantom_net: PhantomData<N>,
}

impl<N> IpTrie<N>
where
    N: IpNet,
{
    fn new() -> IpTrie<N> {
        IpTrie {
            root: None,
            phantom_net: PhantomData,
        }
    }

    fn insert(&mut self, network: N) {
        // The current node
        let mut node = if let Some(root) = &mut self.root {
            if root.is_leaf() {
                // Insert into all-zero network has no effect.
                return;
            }
            root
        } else {
            self.root = Some(IpTrieNode::new());
            self.root.as_mut().unwrap()
        };

        let bits = network.prefix_bits();
        for bit in bits {
            let i = bit as usize;
            let child = &mut node.children[i];
            match child {
                Some(child) => {
                    if child.is_leaf() {
                        // This means the network to be inserted
                        // is already in the trie.
                        return;
                    }
                    node = child;
                }
                None => {
                    *child = Some(Box::new(IpTrieNode::new()));
                    node = child.as_mut().unwrap();
                }
            }
        }
        node.children = [None, None];
    }

    fn search(&self, network: N) -> Option<N> {
        let mut node = self.root.as_ref()?;

        let bits = network.prefix_bits();
        for (j, bit) in bits.enumerate() {
            if node.is_leaf() {
                return Some(network.with_new_prefix(j as u8));
            }

            let i = bit as usize;
            let child = node.children[i].as_ref();
            match child {
                Some(child) => node = child,
                None => return None,
            }
        }

        if node.is_leaf() {
            Some(network)
        } else {
            None
        }

        // The commented code below is more clear. However, this uses a
        // commented method `search` in IpTrieNode, and the performance
        // is relatively poorer that the implementation above.

        // self.root.as_ref().and_then(|root| {
        //     let mut bits = network.prefix_bits();
        //     let first_bit = bits.next();
        //     root.borrow()
        //         .search(bits, first_bit, 0)
        //         .map(|prefix_size| {
        //             network.with_new_prefix(prefix_size)
        //         })
        // })
    }

    fn remove(&mut self, network: N) {
        if let Some(root) = self.root.as_mut() {
            let mut bits = network.prefix_bits();
            if let Some(next_bit) = bits.next() {
                root.remove(bits, next_bit);
                // If root becomes a leaf after removing the network,
                // we should simply reinitialize the trie.
                if !root.is_leaf() {
                    return;
                }
            }
        }
        self.root = None // Reinitialize the trie
    }

    fn simplify(&mut self) {
        if let Some(root) = self.root.as_mut() {
            root.simplify();
        }
    }
}

/// Node of the inner radix trie.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct IpTrieNode {
    children: [Option<Box<IpTrieNode>>; 2],
}

impl IpTrieNode {
    fn new() -> IpTrieNode {
        IpTrieNode {
            children: [None, None],
        }
    }

    #[inline]
    fn init_traverse_state<S: TraverseState>(&self) -> S {
        S::init(self)
    }

    // If both the zero child and the one child of a node are None,
    // it is a leaf node, and it represents a network whose
    // prefix is the path from root to it.
    #[inline]
    fn is_leaf(&self) -> bool {
        self.children[0].is_none() && self.children[1].is_none()
    }

    // If the two children of a node are all leaf node,
    // they can be merged into a new leaf node.
    fn simplify(&mut self) {
        let leaf_count: u32 = self
            .children
            .iter_mut()
            .map(|child| {
                child
                    .as_mut()
                    .map(|child| {
                        child.simplify();
                        child.is_leaf() as u32
                    })
                    .unwrap_or_default()
            })
            .sum();
        if leaf_count == 2 {
            self.children = [None, None];
        }
    }

    //    fn search<I>(&self, mut bits: I, current_bit: Option<bool>, acc: u8) -> Option<u8>
    //        where I: Iterator<Item=bool>
    //    {
    //        if self.is_leaf() {
    //            Some(acc)
    //        } else {
    //            if let Some(current_bit) = current_bit {
    //                if let Some(child) = self.children[current_bit as usize].clone() {
    //                    let next_bit = bits.next();
    //                    return child
    //                        .borrow_mut()
    //                        .search(bits, next_bit, acc + 1);
    //                }
    //            }
    //            None
    //        }
    //    }

    fn remove<I>(&mut self, mut bits: I, current_bit: bool)
    where
        I: Iterator<Item = bool>,
    {
        let i = current_bit as usize;
        let next_bit = bits.next();

        // If the current node is a leaf node, and we have a network
        // to remove, we must split it into two deeper nodes.
        if self.is_leaf() {
            self.children = [
                Some(Box::new(IpTrieNode::new())),
                Some(Box::new(IpTrieNode::new())),
            ];
        }

        match next_bit {
            Some(next_bit) => {
                let is_leaf = if let Some(child) = self.children[i].as_mut() {
                    // Remove the deeper node recursively
                    child.remove(bits, next_bit);
                    child.is_leaf()
                } else {
                    false
                };
                // In general, a leaf node represents a complete
                // network. However, the child node cannot be a complete
                // network after removing a network from it.
                // This occurring indicates the only child of the
                // child node is removed, and now this child node
                // should be marked None.
                if is_leaf {
                    self.children[i] = None;
                }
            }
            None => {
                // Remove the node that represents the network.
                self.children[i] = None;
            }
        }
    }
}

#[cfg(any(feature = "ipnet", feature = "ipnetwork"))]
const MSO_U128: u128 = 1 << 127; // Most significant one for u128
#[cfg(any(feature = "ipnet", feature = "ipnetwork"))]
const MSO_U32: u32 = 1 << 31; // Most significant one for u32
