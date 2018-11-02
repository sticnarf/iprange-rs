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
//! extern crate iprange;
//! extern crate ipnet;
//!
//! use std::net::Ipv4Addr;
//! use iprange::IpRange;
//! use ipnet::Ipv4Net;
//!
//! fn main() {
//!     let ip_range: IpRange<Ipv4Net> = ["10.0.0.0/8", "172.16.0.0/16", "192.168.1.0/24"]
//!         .iter()
//!         .map(|s| s.parse().unwrap())
//!         .collect();
//!
//!     assert!(ip_range.contains(&"172.16.32.1".parse::<Ipv4Addr>().unwrap()));
//!     assert!(ip_range.contains(&"192.168.1.1".parse::<Ipv4Addr>().unwrap()));
//! }
//! ```
//!
//! [`IpRange`]: struct.IpRange.html
//! [`IpNet`]: trait.IpNet.html
//! [`Ipv4Net`]: https://docs.rs/ipnet/1.0.0/ipnet/struct.Ipv4Net.html
//! [`merge`]: struct.IpRange.html#method.merge
//! [`intersect`]: struct.IpRange.html#method.intersect
//! [`exclude`]: struct.IpRange.html#method.exclude

extern crate ipnet;

use ipnet::{Ipv4Net, Ipv6Net};
use std::collections::VecDeque;
use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::net::{Ipv4Addr, Ipv6Addr};

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
/// extern crate ipnet;
/// extern crate iprange;
///
/// use iprange::IpRange;
/// use ipnet::Ipv4Net;
///
/// fn main() {
///     let ip_range: IpRange<Ipv4Net> = ["172.16.0.0/16", "192.168.1.0/24"]
///         .iter()
///         .map(|s| s.parse().unwrap())
///         .collect();
///
///     for network in &ip_range {
///         println!("{:?}", network);
///     }
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
    /// extern crate iprange;
    /// extern crate ipnet;
    ///
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// fn main() {
    ///     let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    ///     ip_range.add("192.168.0.0/24".parse().unwrap())
    ///            .add("192.168.1.0/24".parse().unwrap());
    ///     assert_eq!(ip_range.into_iter().count(), 2);
    ///
    ///     ip_range.simplify();
    ///     assert_eq!(ip_range.into_iter().count(), 1);
    /// }
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
    /// extern crate iprange;
    /// extern crate ipnet;
    ///
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// fn main() {
    ///     let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    ///     ip_range.add("192.168.0.0/23".parse().unwrap())
    ///             .remove("192.168.0.0/24".parse().unwrap());
    ///     // Now, ip_range has only one network: "192.168.1.0/24".
    /// }
    /// ```
    pub fn remove(&mut self, network: N) -> &mut Self {
        self.trie.remove(network);
        self
    }

    /// Simplify `self` by combining networks. For example:
    ///
    /// ```
    /// extern crate iprange;
    /// extern crate ipnet;
    ///
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// fn main() {
    ///     let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    ///     ip_range
    ///         .add("192.168.0.0/20".parse().unwrap())
    ///         .add("192.168.16.0/22".parse().unwrap())
    ///         .add("192.168.20.0/24".parse().unwrap())
    ///         .add("192.168.21.0/24".parse().unwrap())
    ///         .add("192.168.22.0/24".parse().unwrap())
    ///         .add("192.168.23.0/24".parse().unwrap())
    ///         .add("192.168.24.0/21".parse().unwrap())
    ///         .simplify();
    ///     // Now, ip_range has only one network: "192.168.0.0/19".
    /// }
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
    type IntoIter = IpRangeIter<N>;

    fn into_iter(self) -> Self::IntoIter {
        let mut queue = VecDeque::new();
        if let Some(root) = self.trie.root.as_ref() {
            let state: N::S = root.init_traverse_state();
            queue.push_back(state);
        }
        IpRangeIter { queue }
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
pub struct IpRangeIter<N>
where
    N: IpNet,
{
    queue: VecDeque<N::S>,
}

/// Used for internal traversing.
///
/// You can simply ignore this trait.
pub trait TraverseState {
    type Net: IpNet;

    fn node(&self) -> &IpTrieNode;

    fn init(root: &IpTrieNode) -> Self;

    fn transit(&self, next_node: &IpTrieNode, current_bit: bool) -> Self;

    fn build(&self) -> Self::Net;
}

impl<N> Iterator for IpRangeIter<N>
where
    N: IpNet,
{
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(elem) = self.queue.pop_front() {
            // Get the front element of the queue.
            // If it is a leaf, it represents a network
            let node = elem.node();
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
        if self.root.is_none() {
            self.root = Some(IpTrieNode::new())
        }

        let mut node = self.root.as_mut().unwrap() as *mut IpTrieNode; // The current node

        unsafe {
            let bits = network.prefix_bits();
            for bit in bits {
                let i = bit as usize;
                let child = &mut (*node).children[i];
                match child {
                    Some(child) => {
                        if child.is_leaf() {
                            // This means the network to be inserted
                            // is already in the trie.
                            return;
                        }
                        node = &mut **child as *mut IpTrieNode;
                    }
                    None => {
                        (*node).children[i] = Some(Box::new(IpTrieNode::new()));
                        node = (*node).children[i].as_mut().unwrap().as_mut() as *mut IpTrieNode;
                    }
                }
            }
            (*node).children = [None, None];
        }
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
                return;
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
                    }).unwrap_or_default()
            }).sum();
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

const MSO_U128: u128 = 1 << 127; // Most significant one for u128
const MSO_U32: u32 = 1 << 31; // Most significant one for u32

impl IpNet for Ipv4Net {
    type S = Ipv4TraverseState;
    type I = Ipv4PrefixBitIterator;

    #[inline]
    fn prefix_bits(&self) -> Self::I {
        let prefix: u32 = self.addr().into();
        Ipv4PrefixBitIterator {
            prefix,
            prefix_len: self.prefix_len(),
        }
    }

    #[inline]
    fn prefix_len(&self) -> u8 {
        self.prefix_len()
    }

    #[inline]
    fn with_new_prefix(&self, len: u8) -> Self {
        Ipv4Net::new(self.addr(), len).unwrap().trunc()
    }
}

impl ToNetwork<Ipv4Net> for Ipv4Net {
    #[inline]
    fn to_network(&self) -> Ipv4Net {
        self.trunc()
    }
}

impl ToNetwork<Ipv4Net> for Ipv4Addr {
    #[inline]
    fn to_network(&self) -> Ipv4Net {
        Ipv4Net::new(*self, 32).unwrap()
    }
}

impl ToNetwork<Ipv4Net> for u32 {
    #[inline]
    fn to_network(&self) -> Ipv4Net {
        Ipv4Net::new((*self).into(), 32).unwrap()
    }
}

impl ToNetwork<Ipv4Net> for [u8; 4] {
    #[inline]
    fn to_network(&self) -> Ipv4Net {
        Ipv4Net::new((*self).into(), 32).unwrap()
    }
}

pub struct Ipv4TraverseState {
    node: *const IpTrieNode,
    prefix: u32,
    prefix_len: u8,
}

impl TraverseState for Ipv4TraverseState {
    type Net = Ipv4Net;

    #[inline]
    fn node(&self) -> &IpTrieNode {
        unsafe { &*self.node }
    }

    #[inline]
    fn init(root: &IpTrieNode) -> Self {
        Ipv4TraverseState {
            node: root,
            prefix: 0,
            prefix_len: 0,
        }
    }

    #[inline]
    fn transit(&self, next_node: &IpTrieNode, current_bit: bool) -> Self {
        let mask = if current_bit {
            MSO_U32 >> self.prefix_len
        } else {
            0
        };
        Ipv4TraverseState {
            node: next_node,
            prefix: self.prefix | mask,
            prefix_len: self.prefix_len + 1,
        }
    }

    #[inline]
    fn build(&self) -> Self::Net {
        Ipv4Net::new(self.prefix.into(), self.prefix_len as u8).unwrap()
    }
}

pub struct Ipv4PrefixBitIterator {
    prefix: u32,
    prefix_len: u8,
}

impl Iterator for Ipv4PrefixBitIterator {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.prefix_len > 0 {
            let prefix = self.prefix;
            self.prefix <<= 1;
            self.prefix_len -= 1;
            Some(prefix & MSO_U32 != 0)
        } else {
            None
        }
    }
}

impl IpNet for Ipv6Net {
    type S = Ipv6TraverseState;
    type I = Ipv6PrefixBitIterator;

    #[inline]
    fn prefix_bits(&self) -> Self::I {
        Ipv6PrefixBitIterator {
            prefix: self.addr().into(),
            prefix_len: self.prefix_len(),
        }
    }

    #[inline]
    fn prefix_len(&self) -> u8 {
        self.prefix_len()
    }

    #[inline]
    fn with_new_prefix(&self, len: u8) -> Self {
        Ipv6Net::new(self.addr(), len).unwrap().trunc()
    }
}

impl ToNetwork<Ipv6Net> for Ipv6Net {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        self.trunc()
    }
}

impl ToNetwork<Ipv6Net> for Ipv6Addr {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        Ipv6Net::new(*self, 128).unwrap()
    }
}

impl ToNetwork<Ipv6Net> for u128 {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        Ipv6Net::new((*self).into(), 128).unwrap()
    }
}

impl ToNetwork<Ipv6Net> for [u8; 16] {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        Ipv6Net::new((*self).into(), 128).unwrap()
    }
}

impl ToNetwork<Ipv6Net> for [u16; 8] {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        Ipv6Net::new((*self).into(), 128).unwrap()
    }
}

pub struct Ipv6TraverseState {
    node: *const IpTrieNode,
    prefix: u128,
    prefix_len: u8,
}

impl TraverseState for Ipv6TraverseState {
    type Net = Ipv6Net;

    #[inline]
    fn node(&self) -> &IpTrieNode {
        unsafe { &*self.node }
    }

    #[inline]
    fn init(root: &IpTrieNode) -> Self {
        Ipv6TraverseState {
            node: root,
            prefix: 0,
            prefix_len: 0,
        }
    }

    #[inline]
    fn transit(&self, next_node: &IpTrieNode, current_bit: bool) -> Self {
        let mask = if current_bit {
            MSO_U128 >> self.prefix_len
        } else {
            0
        };
        Ipv6TraverseState {
            node: next_node,
            prefix: self.prefix | mask,
            prefix_len: self.prefix_len + 1,
        }
    }

    #[inline]
    fn build(&self) -> Self::Net {
        Ipv6Net::new(self.prefix.into(), self.prefix_len as u8).unwrap()
    }
}

pub struct Ipv6PrefixBitIterator {
    prefix: u128,
    prefix_len: u8,
}

impl Iterator for Ipv6PrefixBitIterator {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.prefix_len > 0 {
            let prefix = self.prefix;
            self.prefix <<= 1;
            self.prefix_len -= 1;
            Some(prefix & MSO_U128 != 0)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_invalid_networks() {
        assert!("192.168.256.130/5".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.130/-1".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.130/33".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.33".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.130/0.0.0".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.130/0.0.0.256".parse::<Ipv4Net>().is_err());
    }

    impl IpRange<Ipv4Net> {
        fn get_network(&self, prefix_size: usize, prefix: &str) -> Option<Ipv4Net> {
            self.trie
                .search(format!("{}/{}", prefix, prefix_size).parse().unwrap())
        }
    }

    #[test]
    fn add_single_network() {
        let mut ip_range = IpRange::new();
        let network = "192.168.5.0/24".parse().unwrap();
        ip_range.add(network);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network), ip_range.get_network(24, "192.168.5.0"));
    }

    #[test]
    fn add_multiple_networks_disjoint() {
        let mut ip_range = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range
            .add(network1)
            .add(network2)
            .add(network3)
            .add(network4)
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(network1), ip_range.get_network(8, "10.0.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
        assert_eq!(Some(network3), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(Some(network4), ip_range.get_network(32, "254.254.254.254"));
    }

    #[test]
    fn simplify() {
        let mut ip_range = IpRange::new();
        ip_range
            .add("192.168.0.0/20".parse().unwrap())
            .add("192.168.16.0/22".parse().unwrap())
            .add("192.168.20.0/24".parse().unwrap())
            .add("192.168.21.0/24".parse().unwrap())
            .add("192.168.22.0/24".parse().unwrap())
            .add("192.168.23.0/24".parse().unwrap())
            .add("192.168.24.0/21".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            "192.168.0.0/19".parse().ok(),
            ip_range.get_network(19, "192.168.0.0")
        );
    }

    #[test]
    fn add_multiple_networks_joint1() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_networks_joint2() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_networks_joint3() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range.add(network2).add(network1).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_networks_joint4() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.5.0/24".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(24, "172.16.5.0"));
    }

    #[test]
    fn add_multiple_networks_joint5() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
    }

    #[test]
    fn add_multiple_networks_joint6() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "0.0.0.0/0".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(0, "0.0.0.0"));
    }

    #[test]
    fn remove_networks_no_split() {
        let mut ip_range = IpRange::new();
        let network1 = "192.168.0.0/24".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        ip_range.remove(network1);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
    }

    #[test]
    fn remove_networks_split1() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.2.0/23".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            Some("192.168.0.0/23".parse().unwrap()),
            ip_range.get_network(23, "192.168.0.0")
        );
    }

    #[test]
    fn remove_networks_split2() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.0.0/23".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            Some("192.168.2.0/23".parse().unwrap()),
            ip_range.get_network(23, "192.168.2.0")
        );
    }

    #[test]
    fn remove_networks_split3() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.2.0/25".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(
            Some("192.168.0.0/23".parse().unwrap()),
            ip_range.get_network(23, "192.168.0.0")
        );
        assert_eq!(
            Some("192.168.2.128/25".parse().unwrap()),
            ip_range.get_network(25, "192.168.2.128")
        );
        assert_eq!(
            Some("192.168.3.0/24".parse().unwrap()),
            ip_range.get_network(24, "192.168.3.0")
        );
    }

    impl IpRange<Ipv4Net> {
        fn contains_ip(&self, ip: &str) -> bool {
            self.contains(&ip.parse::<Ipv4Addr>().unwrap())
        }

        fn find_network_by_ip(&self, ip: &str) -> Option<Ipv4Net> {
            self.supernet(&ip.parse::<Ipv4Addr>().unwrap())
        }

        fn contains_network(&self, network: &str) -> bool {
            self.contains(&network.parse::<Ipv4Net>().unwrap())
        }

        fn super_network_by_network(&self, network: &str) -> Option<Ipv4Net> {
            self.supernet(&network.parse::<Ipv4Net>().unwrap())
        }
    }

    #[test]
    fn contains_ip_with_one_network() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/24".parse().unwrap());

        assert!(ip_range.contains_ip("192.168.0.0"));
        assert!(ip_range.contains_ip("192.168.0.128"));
        assert!(ip_range.contains_ip("192.168.0.255"));
        assert!(!ip_range.contains_ip("192.167.255.255"));
        assert!(!ip_range.contains_ip("192.168.1.0"));
    }

    #[test]
    fn contains_ip_with_many_networks() {
        let mut ip_range = IpRange::new();
        ip_range
            .add("192.168.0.0/24".parse().unwrap())
            .add("172.16.0.0/16".parse().unwrap())
            .add("10.0.0.0/8".parse().unwrap())
            .simplify();

        assert!(ip_range.contains_ip("192.168.0.128"));
        assert!(ip_range.contains_ip("172.16.32.1"));
        assert!(ip_range.contains_ip("10.10.10.10"));
        assert!(!ip_range.contains_ip("0.0.0.0"));
        assert!(!ip_range.contains_ip("8.8.8.8"));
        assert!(!ip_range.contains_ip("11.0.0.0"));
        assert!(!ip_range.contains_ip("192.167.255.255"));
        assert!(!ip_range.contains_ip("255.255.255.255"));
    }

    #[test]
    fn contains_ip_boundary1() {
        let mut ip_range = IpRange::new();
        ip_range.add("0.0.0.0/0".parse().unwrap());

        assert!(ip_range.contains_ip("0.0.0.0"));
        assert!(ip_range.contains_ip("8.8.8.8"));
        assert!(ip_range.contains_ip("192.168.0.0"));
        assert!(ip_range.contains_ip("192.168.1.1"));
    }

    #[test]
    fn contains_ip_boundary2() {
        let mut ip_range = IpRange::new();
        ip_range.add("254.254.254.254/32".parse().unwrap());

        assert!(!ip_range.contains_ip("0.0.0.0"));
        assert!(!ip_range.contains_ip("8.8.8.8"));
        assert!(!ip_range.contains_ip("192.168.0.0"));
        assert!(ip_range.contains_ip("254.254.254.254"));
    }

    #[test]
    fn find_network_with_one_network() {
        let mut ip_range = IpRange::new();
        let network = "192.168.0.0/24".parse().unwrap();
        ip_range.add(network);

        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.0.0"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.0.128"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.0.255"));
        assert_eq!(None, ip_range.find_network_by_ip("192.167.255.255"));
        assert_eq!(None, ip_range.find_network_by_ip("192.168.1.0"));
    }

    #[test]
    fn find_network_with_many_networks() {
        let mut ip_range = IpRange::new();
        let network1 = "192.168.0.0/24".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "10.0.0.0/8".parse().unwrap();
        ip_range
            .add(network1)
            .add(network2)
            .add(network3)
            .simplify();

        assert_eq!(Some(network1), ip_range.find_network_by_ip("192.168.0.128"));
        assert_eq!(Some(network2), ip_range.find_network_by_ip("172.16.32.1"));
        assert_eq!(Some(network3), ip_range.find_network_by_ip("10.10.10.10"));
        assert_eq!(None, ip_range.find_network_by_ip("0.0.0.0"));
        assert_eq!(None, ip_range.find_network_by_ip("8.8.8.8"));
        assert_eq!(None, ip_range.find_network_by_ip("11.0.0.0"));
        assert_eq!(None, ip_range.find_network_by_ip("192.167.255.255"));
        assert_eq!(None, ip_range.find_network_by_ip("255.255.255.255"));
    }

    #[test]
    fn find_network_boundary1() {
        let mut ip_range = IpRange::new();
        let network = "0.0.0.0/0".parse().unwrap();
        ip_range.add(network);

        assert_eq!(Some(network), ip_range.find_network_by_ip("0.0.0.0"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("8.8.8.8"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.0.0"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.1.1"));
    }

    #[test]
    fn find_network_boundary2() {
        let mut ip_range = IpRange::new();
        let network = "254.254.254.254/32".parse().unwrap();
        ip_range.add(network);

        assert_eq!(None, ip_range.find_network_by_ip("0.0.0.0"));
        assert_eq!(None, ip_range.find_network_by_ip("8.8.8.8"));
        assert_eq!(None, ip_range.find_network_by_ip("192.168.0.0"));
        assert_eq!(
            Some(network),
            ip_range.find_network_by_ip("254.254.254.254")
        );
    }

    #[test]
    fn contains_network_with_one_network() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/24".parse().unwrap());

        assert!(ip_range.contains_network("192.168.0.0/24"));
        assert!(ip_range.contains_network("192.168.0.128/25"));
        assert!(!ip_range.contains_network("192.168.0.0/23"));
        assert!(!ip_range.contains_network("192.168.1.0/24"));
        assert!(!ip_range.contains_network("192.167.0.0/24"));
    }

    #[test]
    fn contains_network_with_many_networks() {
        let mut ip_range = IpRange::new();
        ip_range
            .add("192.168.0.0/24".parse().unwrap())
            .add("172.16.0.0/16".parse().unwrap())
            .add("10.0.0.0/8".parse().unwrap())
            .simplify();

        assert!(ip_range.contains_network("192.168.0.128/25"));
        assert!(ip_range.contains_network("172.16.32.0/20"));
        assert!(ip_range.contains_network("10.10.0.0/16"));
        assert!(!ip_range.contains_network("0.0.0.0/0"));
        assert!(!ip_range.contains_network("8.0.0.0/6"));
        assert!(!ip_range.contains_network("8.0.0.0/7"));
        assert!(!ip_range.contains_network("11.0.0.0/9"));
        assert!(!ip_range.contains_network("192.167.255.255/32"));
        assert!(!ip_range.contains_network("255.0.0.0/8"));
    }

    #[test]
    fn contains_network_boundary1() {
        let mut ip_range = IpRange::new();
        ip_range.add("0.0.0.0/0".parse().unwrap());

        assert!(ip_range.contains_network("0.0.0.0/0"));
        assert!(ip_range.contains_network("8.0.0.0/6"));
        assert!(ip_range.contains_network("11.0.0.0/9"));
        assert!(ip_range.contains_network("192.168.0.128/25"));
        assert!(ip_range.contains_network("255.255.255.255/32"));
    }

    #[test]
    fn contains_network_boundary2() {
        let mut ip_range = IpRange::new();
        ip_range.add("254.254.254.254/32".parse().unwrap());

        assert!(!ip_range.contains_network("0.0.0.0/0"));
        assert!(!ip_range.contains_network("8.0.0.0/6"));
        assert!(!ip_range.contains_network("254.254.0.0/16"));
        assert!(ip_range.contains_network("254.254.254.254/32"));
        assert!(!ip_range.contains_network("255.255.255.255/32"));
    }

    #[test]
    fn super_network_with_one_network() {
        let mut ip_range = IpRange::new();
        let network = "192.168.0.0/24".parse().unwrap();
        ip_range.add(network);

        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("192.168.0.0/24")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("192.168.0.128/25")
        );
        assert_eq!(None, ip_range.super_network_by_network("192.168.0.0/23"));
        assert_eq!(None, ip_range.super_network_by_network("192.168.1.0/24"));
        assert_eq!(None, ip_range.super_network_by_network("192.167.0.0/24"));
    }

    #[test]
    fn super_network_with_many_networks() {
        let mut ip_range = IpRange::new();
        let network1 = "192.168.0.0/24".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "10.0.0.0/8".parse().unwrap();
        ip_range
            .add(network1)
            .add(network2)
            .add(network3)
            .simplify();

        assert_eq!(
            Some(network1),
            ip_range.super_network_by_network("192.168.0.128/25")
        );
        assert_eq!(
            Some(network2),
            ip_range.super_network_by_network("172.16.32.0/20")
        );
        assert_eq!(
            Some(network3),
            ip_range.super_network_by_network("10.10.0.0/16")
        );
        assert_eq!(None, ip_range.super_network_by_network("0.0.0.0/0"));
        assert_eq!(None, ip_range.super_network_by_network("8.0.0.0/6"));
        assert_eq!(None, ip_range.super_network_by_network("8.0.0.0/7"));
        assert_eq!(None, ip_range.super_network_by_network("11.0.0.0/9"));
        assert_eq!(
            None,
            ip_range.super_network_by_network("192.167.255.255/32")
        );
        assert_eq!(None, ip_range.super_network_by_network("255.0.0.0/8"));
    }

    #[test]
    fn super_network_boundary1() {
        let mut ip_range = IpRange::new();
        let network = "0.0.0.0/0".parse().unwrap();
        ip_range.add(network);

        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("0.0.0.0/0")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("8.0.0.0/6")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("11.0.0.0/9")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("192.168.0.128/25")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("255.255.255.255/32")
        );
    }

    #[test]
    fn super_network_boundary2() {
        let mut ip_range = IpRange::new();
        let network = "254.254.254.254/32".parse().unwrap();
        ip_range.add(network);

        assert_eq!(None, ip_range.super_network_by_network("0.0.0.0/0"));
        assert_eq!(None, ip_range.super_network_by_network("8.0.0.0/6"));
        assert_eq!(None, ip_range.super_network_by_network("254.254.0.0/16"));
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("254.254.254.254/32")
        );
        assert_eq!(
            None,
            ip_range.super_network_by_network("255.255.255.255/32")
        );
    }

    #[test]
    fn merge_empty1() {
        let ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range2
            .add(network1)
            .add(network2)
            .add(network3)
            .add(network4)
            .simplify();

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(network1), ip_range.get_network(8, "10.0.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
        assert_eq!(Some(network3), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(Some(network4), ip_range.get_network(32, "254.254.254.254"));
    }

    #[test]
    fn merge_empty2() {
        let mut ip_range1 = IpRange::new();
        let ip_range2 = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range1
            .add(network1)
            .add(network2)
            .add(network3)
            .add(network4)
            .simplify();

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(network1), ip_range.get_network(8, "10.0.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
        assert_eq!(Some(network3), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(Some(network4), ip_range.get_network(32, "254.254.254.254"));
    }

    #[test]
    fn merge_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(network1), ip_range.get_network(8, "10.0.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
        assert_eq!(Some(network3), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(Some(network4), ip_range.get_network(32, "254.254.254.254"));
    }

    #[test]
    fn merge_joint1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn merge_joint2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn merge_sequent1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.5.0/24".parse().unwrap();
        let network3 = "172.16.6.0/24".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);
        ip_range2.add(network3);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 2);
        assert_eq!(
            "172.16.4.0/23".parse().ok(),
            ip_range.get_network(23, "172.16.4.0")
        );
        assert_eq!(
            "172.16.6.0/24".parse().ok(),
            ip_range.get_network(24, "172.16.6.0")
        );
    }

    #[test]
    fn merge_sequent2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let mut ip_range3 = IpRange::new();
        ip_range1
            .add("192.168.0.0/20".parse().unwrap())
            .add("192.168.24.0/21".parse().unwrap());
        ip_range2
            .add("192.168.16.0/22".parse().unwrap())
            .add("192.168.23.0/24".parse().unwrap());
        ip_range3
            .add("192.168.20.0/24".parse().unwrap())
            .add("192.168.21.0/24".parse().unwrap())
            .add("192.168.22.0/24".parse().unwrap());

        let ip_range = ip_range1.merge(&ip_range2);
        let ip_range = ip_range.merge(&ip_range3);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            "192.168.0.0/19".parse().ok(),
            ip_range.get_network(19, "192.168.0.0")
        );
    }

    #[test]
    fn intersect_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 0);
    }

    #[test]
    fn intersect_joint1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network1), ip_range.get_network(24, "172.16.4.0"));
    }

    #[test]
    fn intersect_joint2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network1), ip_range.get_network(24, "172.16.5.0"));
    }

    #[test]
    fn intersect_joint3() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.5.0/24".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network1), ip_range.get_network(24, "172.16.5.0"));
    }

    #[test]
    fn intersect_joint4() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "192.168.0.0/24".parse().unwrap();
        let network3 = "10.10.0.0/16".parse().unwrap();
        let network4 = "10.254.0.0/17".parse().unwrap();
        let network5 = "192.168.0.0/16".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4).add(network5);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(network3), ip_range.get_network(16, "10.10.0.0"));
        assert_eq!(Some(network4), ip_range.get_network(17, "10.254.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.0.0"));
    }

    #[test]
    fn exclude_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range1, ip_range);
    }

    #[test]
    fn exclude_larger() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "172.16.4.0/24".parse().unwrap();
        let network2 = "192.168.1.0/24".parse().unwrap();
        let network3 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.1.0"));
    }

    #[test]
    fn exclude_identical() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "172.16.5.0/24".parse().unwrap();
        let network2 = "192.168.1.0/24".parse().unwrap();
        let network3 = "172.16.4.0/22".parse().unwrap();
        let network4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.1.0"));
    }

    #[test]
    fn exclude_split1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "172.16.4.0/22".parse().unwrap();
        let network2 = "192.168.1.0/24".parse().unwrap();
        let network3 = "172.16.5.0/24".parse().unwrap();
        let network4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(
            "172.16.4.0/24".parse().ok(),
            ip_range.get_network(24, "172.16.4.0")
        );
        assert_eq!(
            "172.16.6.0/23".parse().ok(),
            ip_range.get_network(23, "172.16.6.0")
        );
    }

    #[test]
    fn exclude_split2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "172.16.4.0/22".parse().unwrap();
        let network2 = "192.168.1.0/24".parse().unwrap();
        let network3 = "172.16.4.0/24".parse().unwrap();
        let network4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(
            "172.16.5.0/24".parse().ok(),
            ip_range.get_network(24, "172.16.5.0")
        );
        assert_eq!(
            "172.16.6.0/23".parse().ok(),
            ip_range.get_network(23, "172.16.6.0")
        );
    }

    #[test]
    fn iter_ipv4() {
        let mut data = vec!["1.0.1.0/24", "1.0.2.0/23", "1.0.8.0/21"];
        let ip_range: IpRange<Ipv4Net> = data.iter().map(|net| net.parse().unwrap()).collect();
        let mut nets: Vec<String> = ip_range.iter().map(|net| format!("{}", net)).collect();
        data.sort_unstable();
        nets.sort_unstable();
        assert_eq!(nets, data);
    }

    #[test]
    fn iter_ipv6() {
        let mut data = vec![
            "2400:9a40::/32",
            "2400:9dc0::/32",
            "2400:9e00::/32",
            "2400:a040::/32",
        ];
        let ip_range: IpRange<Ipv6Net> = data.iter().map(|net| net.parse().unwrap()).collect();
        let mut nets: Vec<String> = ip_range.iter().map(|net| format!("{}", net)).collect();
        data.sort_unstable();
        nets.sort_unstable();
        assert_eq!(nets, data);
    }

    #[test]
    fn debug_fmt() {
        let ip_range: IpRange<Ipv4Net> = IpRange::default();
        assert_eq!(format!("{:?}", ip_range), "IpRange []");

        let ip_range: IpRange<Ipv4Net> = ["1.0.1.0/24", "1.0.2.0/23", "1.0.8.0/21"]
            .iter()
            .map(|net| net.parse().unwrap())
            .collect();
        assert_eq!(
            format!("{:?}", ip_range),
            "IpRange [1.0.8.0/21, 1.0.2.0/23, 1.0.1.0/24]"
        );

        let ip_range: IpRange<Ipv4Net> = [
            "192.168.0.0/16",
            "1.0.2.0/23",
            "1.0.8.0/21",
            "127.0.0.0/8",
            "172.16.0.0/12",
        ]
            .iter()
            .map(|net| net.parse().unwrap())
            .collect();
        assert_eq!(
            format!("{:?}", ip_range),
            "IpRange [127.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, ...]"
        );

        let ip_range: IpRange<Ipv6Net> = [
            "2001:4438::/32",
            "2001:4510::/29",
            "2400:1040::/32",
            "2400:12c0::/32",
            "2400:1340::/32",
            "2400:1380::/32",
            "2400:15c0::/32",
        ]
            .iter()
            .map(|net| net.parse().unwrap())
            .collect();
        assert_eq!(
            format!("{:?}", ip_range),
            "IpRange [2001:4510::/29, 2001:4438::/32, 2400:1040::/32, ...]"
        );
    }
}
