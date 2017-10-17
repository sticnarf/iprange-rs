//! `iprange` is a library for managing IP ranges.
//!
//! An [`IpRange`] is a set of subnets.
//! You can add or remove a [`Subnet`] from an [`IpRange`].
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
//!
//! use std::net::Ipv4Addr;
//! use iprange::IpRange;
//!
//! fn main() {
//!     let mut ip_range = IpRange::new();
//!     ip_range
//!         .add("10.0.0.0/8".parse().unwrap())
//!         .add("172.16.0.0/16".parse().unwrap())
//!         .add("192.168.1.0/24".parse().unwrap());
//!
//!     assert!(ip_range.contains("172.16.32.1".parse::<Ipv4Addr>().unwrap()));
//!     assert!(ip_range.contains("192.168.1.1".parse::<Ipv4Addr>().unwrap()));
//! }
//! ```
//!
//! Currently, this library supports IPv4 only.
//!
//! [`IpRange`]: struct.IpRange.html
//! [`Subnet`]: struct.Subnet.html
//! [`merge`]: struct.IpRange.html#method.merge
//! [`intersect`]: struct.IpRange.html#method.intersect
//! [`exclude`]: struct.IpRange.html#method.exclude

extern crate byteorder;

use std::str::FromStr;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::net::Ipv4Addr;
use std::rc::Rc;
use std::cell::RefCell;
use std::iter::FromIterator;
use std::collections::VecDeque;
use byteorder::{BigEndian, ByteOrder};

/// A set of subnets that supports various operations.
///
/// `IntoIter` is implemented for `&IpRange`. So, you can use `for`
/// to iterate over the subnets in an `IpRange`:
///
/// ```
/// use iprange::{IpRange, Subnet};
///
/// let ip_range: IpRange = ["172.16.0.0/16", "192.168.1.0/24"]
///     .iter()
///     .map(|s| s.parse::<Subnet>().unwrap())
///     .collect();
///
/// for subnet in &ip_range {
///     println!("{:?}", subnet);
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IpRange {
    // The index of the Vec is the prefix size of the subnets
    // which the corresponding BTreeMap contains
    trie: IpTrie,
}

impl IpRange {
    /// Creates an empty `IpRange`.
    pub fn new() -> IpRange {
        IpRange {
            trie: IpTrie::new(),
        }
    }

    /// Add a subnet to `self`.
    ///
    /// Returns `&mut self` in order to enable method chaining.
    ///
    /// Pay attention that this operation will not combine two
    /// subnets automatically. To do this, call [`simplify`] method
    /// explicitly. For example:
    ///
    /// ```
    /// use iprange::IpRange;
    ///
    /// let mut ip_range = IpRange::new();
    /// ip_range.add("192.168.0.0/24".parse().unwrap())
    ///         .add("192.168.1.0/24".parse().unwrap());
    /// assert_eq!(ip_range.into_iter().count(), 2);
    ///
    /// ip_range.simplify();
    /// assert_eq!(ip_range.into_iter().count(), 1);
    /// ```
    ///
    /// [`simplify`]: struct.IpRange.html#method.simplify
    pub fn add(&mut self, subnet: Subnet) -> &mut IpRange {
        // if !self.includes(subnet) {
        //     self.remove_inside(subnet);
        //     self.subnets[subnet.prefix_size].insert(subnet.prefix);
        // }
        self.trie.insert(subnet);
        self
    }

    /// Remove a subnet from `self`.
    ///
    /// Returns `&mut self` in order to enable method chaining.
    ///
    /// `self` does not necessarily has exactly the subnet to be removed.
    /// The subnet can be a subnet of a subnet in `self`.
    /// This method will do splitting and remove the corresponding subnet.
    /// For example:
    ///
    /// ```
    /// use iprange::IpRange;
    ///
    /// let mut ip_range = IpRange::new();
    /// ip_range.add("192.168.0.0/23".parse().unwrap())
    ///         .remove("192.168.0.0/24".parse().unwrap());
    /// // Now, ip_range has only one subnet: "192.168.1.0/24".
    /// ```
    pub fn remove(&mut self, subnet: Subnet) -> &mut IpRange {
        // self.remove_inside(subnet);

        // while let Some(super_subnet) = self.super_subnet(subnet) {
        //     self.split_subnet(super_subnet);
        //     self.remove_subnet(subnet);
        // }
        self.trie.remove(subnet);
        self
    }

    /// Simplify `self` by combining subnets. For example:
    ///
    /// ```
    /// use iprange::IpRange;
    ///
    /// let mut ip_range = IpRange::new();
    /// ip_range
    ///     .add("192.168.0.0/20".parse().unwrap())
    ///     .add("192.168.16.0/22".parse().unwrap())
    ///     .add("192.168.20.0/24".parse().unwrap())
    ///     .add("192.168.21.0/24".parse().unwrap())
    ///     .add("192.168.22.0/24".parse().unwrap())
    ///     .add("192.168.23.0/24".parse().unwrap())
    ///     .add("192.168.24.0/21".parse().unwrap())
    ///     .simplify();
    /// // Now, ip_range has only one subnet: "192.168.0.0/19".
    /// ```
    pub fn simplify(&mut self) {
        self.trie.simplify();
    }

    /// Returns a new `IpRange` which contains all subnets
    /// that is either in `self` or in `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn merge(&self, other: &IpRange) -> IpRange {
        self.into_iter().chain(other.into_iter()).collect()
    }

    /// Returns a new `IpRange` which contains all subnets
    /// that is in both `self` and `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn intersect(&self, other: &IpRange) -> IpRange {
        let range1 = self.into_iter().filter(|&subnet| other.includes(subnet));
        let range2 = other.into_iter().filter(|&subnet| self.includes(subnet));
        range1.chain(range2).collect()
    }

    /// Returns a new `IpRange` which contains all subnets
    /// that is in `self` while not in `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn exclude(&self, other: &IpRange) -> IpRange {
        let mut new = self.clone();
        for subnet in other {
            new.remove(subnet);
        }
        new
    }

    /// Tests if `ip` is in `self`.
    pub fn contains<T: Into<CompactIpv4>>(&self, ip: T) -> bool {
        self.find_subnet(ip.into()).is_some()
    }

    /// Tests if `self` includes `subnet`.
    pub fn includes(&self, subnet: Subnet) -> bool {
        self.super_subnet(subnet).is_some()
    }

    /// Returns the subnet in `self` that contains `ip`.
    ///
    /// Returns None if no subnet in `self` contains `ip`.
    pub fn find_subnet<T: Into<CompactIpv4>>(&self, ip: T) -> Option<Subnet> {
        self.super_subnet(ip.into().into())
    }

    /// Returns the subnet in `self` which is the supernetwork of `subnet`.
    ///
    /// Returns None if no subnet in `self` includes `subnet`.
    pub fn super_subnet(&self, subnet: Subnet) -> Option<Subnet> {
        self.trie.search(subnet)
    }
}

impl<'a> IntoIterator for &'a IpRange {
    type Item = Subnet;
    type IntoIter = IpRangeIter;

    fn into_iter(self) -> Self::IntoIter {
        let mut queue = VecDeque::new();
        if let Some(root) = self.trie.root.as_ref() {
            queue.push_back(IpRangeIterElem {
                node: root.clone(),
                prefix: 0,
                prefix_size: 0,
            });
        }
        IpRangeIter { queue }
    }
}

/// An iterator over the subnets in an [`IpRange`].
///
/// [`IpRange`]: struct.IpRange.html
pub struct IpRangeIter {
    queue: VecDeque<IpRangeIterElem>,
}

struct IpRangeIterElem {
    node: Rc<RefCell<IpTrieNode>>,
    prefix: u32,
    prefix_size: usize,
}

impl<'a> Iterator for IpRangeIter {
    type Item = Subnet;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(elem) = self.queue.pop_front() {
            if elem.node.borrow().is_leaf() {
                let subnet = Some(Subnet {
                    prefix: CompactIpv4(elem.prefix),
                    prefix_size: elem.prefix_size,
                });
                return subnet;
            }
            if let Some(one) = elem.node.borrow().one.as_ref() {
                self.queue.push_back(IpRangeIterElem {
                    node: one.clone(),
                    prefix: elem.prefix | (1 << (31 - elem.prefix_size)),
                    prefix_size: elem.prefix_size + 1,
                })
            }
            if let Some(zero) = elem.node.borrow().zero.as_ref() {
                self.queue.push_back(IpRangeIterElem {
                    node: zero.clone(),
                    prefix: elem.prefix,
                    prefix_size: elem.prefix_size + 1,
                })
            }
        }
        None
    }
}

impl FromIterator<Subnet> for IpRange {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Subnet>,
    {
        let mut ip_range = IpRange::new();
        for subnet in iter {
            ip_range.add(subnet);
        }
        ip_range.simplify();
        ip_range
    }
}

impl<'a> FromIterator<&'a Subnet> for IpRange {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a Subnet>,
    {
        let mut ip_range = IpRange::new();
        for subnet in iter {
            ip_range.add(*subnet);
        }
        ip_range.simplify();
        ip_range
    }
}

/// A wrapper of `u32` to represent an IPv4 address.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct CompactIpv4(u32);

impl std::ops::BitAnd for CompactIpv4 {
    type Output = CompactIpv4;

    fn bitand(self, other: CompactIpv4) -> CompactIpv4 {
        CompactIpv4(self.0 & other.0)
    }
}

impl std::ops::BitOr for CompactIpv4 {
    type Output = CompactIpv4;

    fn bitor(self, other: CompactIpv4) -> CompactIpv4 {
        CompactIpv4(self.0 | other.0)
    }
}

impl std::ops::BitXor for CompactIpv4 {
    type Output = CompactIpv4;

    fn bitxor(self, other: CompactIpv4) -> CompactIpv4 {
        CompactIpv4(self.0 ^ other.0)
    }
}

impl std::ops::Not for CompactIpv4 {
    type Output = CompactIpv4;

    fn not(self) -> CompactIpv4 {
        CompactIpv4(!self.0)
    }
}

impl std::ops::BitAndAssign for CompactIpv4 {
    fn bitand_assign(&mut self, other: CompactIpv4) {
        self.0 &= other.0;
    }
}

impl std::ops::BitOrAssign for CompactIpv4 {
    fn bitor_assign(&mut self, other: CompactIpv4) {
        self.0 |= other.0;
    }
}

impl std::ops::BitXorAssign for CompactIpv4 {
    fn bitxor_assign(&mut self, other: CompactIpv4) {
        self.0 ^= other.0;
    }
}

impl From<Ipv4Addr> for CompactIpv4 {
    fn from(addr: Ipv4Addr) -> CompactIpv4 {
        CompactIpv4(BigEndian::read_u32(&addr.octets()))
    }
}

impl From<u32> for CompactIpv4 {
    fn from(addr: u32) -> CompactIpv4 {
        CompactIpv4(addr)
    }
}

impl Into<u32> for CompactIpv4 {
    fn into(self: CompactIpv4) -> u32 {
        self.0
    }
}

impl fmt::Debug for CompactIpv4 {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut bytes = [0u8; 4];
        BigEndian::write_u32(&mut bytes, self.0);
        write!(f, "{}.{}.{}.{}", bytes[0], bytes[1], bytes[2], bytes[3])
    }
}

impl CompactIpv4 {
    /// This method returns the successor of `self`.
    ///
    /// If `self` does not have a successor, that is to say
    /// `self` is `255.255.255.255`, then `None` is returned.
    pub fn successor(self) -> Option<CompactIpv4> {
        self.0.checked_add(1).map(|v| CompactIpv4(v))
    }

    /// This method returns the predecessor of `self`.
    ///
    /// If `self` does not have a predecessor, that is to say
    /// `self` is `0.0.0.0`, then `None` is returned.
    pub fn predecessor(self) -> Option<CompactIpv4> {
        self.0.checked_sub(1).map(|v| CompactIpv4(v))
    }
}

/// Represents a subdivision of an IP network.
///
/// We use CIDR notation to repesent a subnetwork.
/// Thus, a `Subnet` is composed with a `prefix` and the size of the subnet
/// written after a slash (`/`). The size of the subnet is
/// represented by `prefix_size`, which is the count of leading ones
/// in the subnet mask.
///
/// `Subnet` implements `FromStr` so that users can easily convert a
/// string slice to a `Subnet`. Currently, two formats are supported:
///
/// * `a.b.c.d/prefix_size` (e.g. `10.0.0.0/8`)
/// * `a.b.c.d/mask` (e.g. `192.168.0.0/255.255.255.0`)
///
/// The subnet will be automatically fixed after parsing. For example:
/// parsing `172.16.32.1/16` results in `172.16.0.0/16`.
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Subnet {
    /// The prefix IP of the subnet
    pub prefix: CompactIpv4,
    /// The count of leading ones of the subnet mask
    pub prefix_size: usize,
}

impl Subnet {
    pub fn new(prefix: CompactIpv4, prefix_size: usize) -> Subnet {
        let mut subnet = Subnet {
            prefix,
            prefix_size,
        };
        subnet.prefix &= subnet.mask();
        subnet
    }

    /// Tests if `self` contains the IP `addr`.
    pub fn contains<T: Into<CompactIpv4>>(&self, addr: T) -> bool {
        (addr.into() & self.mask()) == self.prefix
    }

    /// Returns the last IP in `self`.
    pub fn end(&self) -> CompactIpv4 {
        self.prefix | !(self.mask())
    }

    /// Tests if `self` includes `other`, in other words,
    /// `self` is a supernetwork of `other`.
    pub fn includes(&self, other: Subnet) -> bool {
        self.prefix <= other.prefix && self.end() >= other.end()
    }

    /// Returns the subnet mask of `self`.
    pub fn mask(&self) -> CompactIpv4 {
        CompactIpv4(
            0xffffffffu32
                .checked_shl(32 - self.prefix_size as u32)
                .unwrap_or_default(), // The default value of u32 is 0
        )
    }
}

impl From<CompactIpv4> for Subnet {
    fn from(ip: CompactIpv4) -> Subnet {
        Subnet {
            prefix: ip,
            prefix_size: 32,
        }
    }
}

impl fmt::Debug for Subnet {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}/{}", self.prefix, self.prefix_size)
    }
}

impl FromStr for Subnet {
    type Err = ParseSubnetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.find('/')
            .ok_or(ParseSubnetError::new("No slash found"))
            .and_then(|slash| {
                let prefix = s[..slash]
                    .parse()
                    .map(|addr: Ipv4Addr| addr.into())
                    .map_err(|_| ParseSubnetError::new("Illegal prefix"));

                let mask = s[(slash + 1)..]
                    .parse()
                    .map(|addr: Ipv4Addr| addr.into())
                    .or_else(|_| {
                        s[(slash + 1)..]
                            .parse()
                            .map_err(|_| ParseSubnetError::new("Unknown subnet format"))
                            .and_then(|size: u32| if size == 32 {
                                Ok(0xffffffffu32)
                            } else {
                                0xffffffffu32
                                    .checked_shr(size) // Err when size > 32
                                    .ok_or(ParseSubnetError::new("Prefix size out of range"))
                                    .map(|x| !x)
                            })
                            .map(|mask| CompactIpv4(mask))
                    });

                prefix.and_then(|prefix: CompactIpv4| {
                    mask.map(|mask| {
                        Subnet {
                            prefix: prefix & mask, // Fix subnet
                            prefix_size: (32 - mask.0.trailing_zeros()) as usize,
                        }
                    })
                })
            })
    }
}

/// An error which can be returned when parsing an subnet.
#[derive(Debug)]
pub struct ParseSubnetError {
    reason: String,
}

impl ParseSubnetError {
    fn new(reason: &str) -> ParseSubnetError {
        ParseSubnetError {
            reason: reason.to_owned(),
        }
    }
}

impl Display for ParseSubnetError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "ParseSubnetError: {}", self.reason)
    }
}

impl Error for ParseSubnetError {
    fn description(&self) -> &str {
        &self.reason
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IpTrie {
    root: Option<Rc<RefCell<IpTrieNode>>>,
}

impl IpTrie {
    fn new() -> IpTrie {
        IpTrie { root: None }
    }

    fn insert(&mut self, subnet: Subnet) {
        if self.root.is_none() {
            self.root = Some(Rc::new(RefCell::new(IpTrieNode::new())))
        }

        let mut node = self.root.clone().unwrap();
        let mut tmp_prefix = subnet.prefix.0;
        for _ in 0..subnet.prefix_size {
            let (prefix, overflow) = tmp_prefix.overflowing_mul(2);
            tmp_prefix = prefix;

            let child = if overflow {
                node.borrow().one.clone()
            } else {
                node.borrow().zero.clone()
            };
            match child {
                Some(child) => {
                    if child.borrow().is_leaf() {
                        return;
                    }
                    node = child;
                }
                None => {
                    let new_node = Rc::new(RefCell::new(IpTrieNode::new()));
                    if overflow {
                        (*node.borrow_mut()).one = Some(new_node.clone());
                    } else {
                        (*node.borrow_mut()).zero = Some(new_node.clone());
                    }
                    node = new_node;
                }
            }
        }
        (*node.borrow_mut()).one = None;
        (*node.borrow_mut()).zero = None;
    }

    fn search(&self, subnet: Subnet) -> Option<Subnet> {
        if self.root.is_none() {
            return None;
        }
        let mut node = self.root.clone().unwrap();
        let mut tmp_prefix = subnet.prefix.0;
        for i in 0..subnet.prefix_size {
            if node.borrow().is_leaf() {
                return Some(Subnet::new(subnet.prefix, i));
            }

            let (prefix, overflow) = tmp_prefix.overflowing_mul(2);
            tmp_prefix = prefix;

            let child = if overflow {
                node.borrow().one.clone()
            } else {
                node.borrow().zero.clone()
            };

            match child {
                Some(child) => node = child,
                None => return None,
            }
        }
        if node.borrow().is_leaf() {
            Some(subnet)
        } else {
            None
        }
    }

    fn remove(&mut self, subnet: Subnet) {
        if self.root.is_none() {
            return;
        }

        let node = self.root.clone().unwrap();
        node.borrow_mut()
            .remove(subnet.prefix.0, subnet.prefix_size);
    }

    fn simplify(&mut self) {
        if let Some(root) = self.root.as_ref() {
            root.borrow_mut().simplify();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IpTrieNode {
    zero: Option<Rc<RefCell<IpTrieNode>>>,
    one: Option<Rc<RefCell<IpTrieNode>>>,
}

impl IpTrieNode {
    fn new() -> IpTrieNode {
        IpTrieNode {
            zero: None,
            one: None,
        }
    }

    fn is_leaf(&self) -> bool {
        self.zero.is_none() && self.one.is_none()
    }

    fn simplify(&mut self) {
        let mut leaf_count = 0;
        if let Some(zero) = self.zero.as_ref() {
            zero.borrow_mut().simplify();
            if zero.borrow().is_leaf() {
                leaf_count += 1;
            }
        }
        if let Some(one) = self.one.as_ref() {
            one.borrow_mut().simplify();
            if one.borrow().is_leaf() {
                leaf_count += 1;
            }
        }
        if leaf_count == 2 {
            self.one = None;
            self.zero = None;
        }
    }

    fn remove(&mut self, prefix: u32, prefix_size: usize) {
        let (prefix, overflow) = prefix.overflowing_mul(2);
        if self.is_leaf() {
            self.one = Some(Rc::new(RefCell::new(IpTrieNode::new())));
            self.zero = Some(Rc::new(RefCell::new(IpTrieNode::new())));
        }
        if prefix_size == 1 {
            if overflow {
                self.one = None;
            } else {
                self.zero = None;
            }
            return;
        }
        if overflow {
            if let Some(child) = self.one.clone() {
                child.borrow_mut().remove(prefix, prefix_size - 1);
                if child.borrow().is_leaf() {
                    self.one = None;
                }
            }
        } else {
            if let Some(child) = self.zero.clone() {
                child.borrow_mut().remove(prefix, prefix_size - 1);
                if child.borrow().is_leaf() {
                    self.zero = None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_subnet_cidr() {
        let my_subnet: Subnet = "192.168.5.130/24".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11000000_10101000_00000101_00000000.into(),
                prefix_size: 24,
            }
        );
    }

    #[test]
    fn parse_subnet_with_only_one_ip() {
        let my_subnet: Subnet = "233.233.233.233/32".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11101001_11101001_11101001_11101001.into(),
                prefix_size: 32,
            }
        );
    }

    #[test]
    fn parse_subnet_with_mask() {
        let my_subnet: Subnet = "192.168.5.130/255.255.255.192".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11000000_10101000_00000101_10000000.into(),
                prefix_size: 26,
            }
        );
    }

    #[test]
    fn parse_invalid_subnets() {
        assert!("192.168.256.130/5".parse::<Subnet>().is_err());
        assert!("192.168.5.130/-1".parse::<Subnet>().is_err());
        assert!("192.168.5.130/33".parse::<Subnet>().is_err());
        assert!("192.168.5.33".parse::<Subnet>().is_err());
        assert!("192.168.5.130/0.0.0".parse::<Subnet>().is_err());
        assert!("192.168.5.130/0.0.0.256".parse::<Subnet>().is_err());
    }

    impl IpRange {
        fn get_subnet(&self, prefix_size: usize, prefix: &str) -> Option<Subnet> {
            self.trie.search(Subnet {
                prefix: prefix.parse::<Ipv4Addr>().unwrap().into(),
                prefix_size,
            })
        }
    }

    #[test]
    fn add_single_subnet() {
        let mut ip_range = IpRange::new();
        let subnet = "192.168.5.130/24".parse().unwrap();
        ip_range.add(subnet);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet), ip_range.get_subnet(24, "192.168.5.0"));
    }

    #[test]
    fn add_multiple_subnets_disjoint() {
        let mut ip_range = IpRange::new();
        let subnet1 = "10.0.0.1/8".parse().unwrap();
        let subnet2 = "172.16.5.130/16".parse().unwrap();
        let subnet3 = "192.168.1.1/24".parse().unwrap();
        let subnet4 = "254.254.254.254/32".parse().unwrap();
        ip_range
            .add(subnet1)
            .add(subnet2)
            .add(subnet3)
            .add(subnet4)
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(subnet1), ip_range.get_subnet(8, "10.0.0.0"));
        assert_eq!(Some(subnet2), ip_range.get_subnet(16, "172.16.0.0"));
        assert_eq!(Some(subnet3), ip_range.get_subnet(24, "192.168.1.0"));
        assert_eq!(Some(subnet4), ip_range.get_subnet(32, "254.254.254.254"));
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
            ip_range.get_subnet(19, "192.168.0.0")
        );
    }


    #[test]
    fn add_multiple_subnets_joint1() {
        let mut ip_range = IpRange::new();
        let subnet1 = "172.16.4.130/24".parse().unwrap();
        let subnet2 = "172.16.5.130/22".parse().unwrap();
        ip_range.add(subnet1).add(subnet2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_subnets_joint2() {
        let mut ip_range = IpRange::new();
        let subnet1 = "172.16.5.130/24".parse().unwrap();
        let subnet2 = "172.16.4.130/22".parse().unwrap();
        ip_range.add(subnet1).add(subnet2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_subnets_joint3() {
        let mut ip_range = IpRange::new();
        let subnet1 = "172.16.4.130/24".parse().unwrap();
        let subnet2 = "172.16.5.130/22".parse().unwrap();
        ip_range.add(subnet2).add(subnet1).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_subnets_joint4() {
        let mut ip_range = IpRange::new();
        let subnet1 = "172.16.5.130/24".parse().unwrap();
        let subnet2 = "172.16.5.131/24".parse().unwrap();
        ip_range.add(subnet1).add(subnet2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(24, "172.16.5.0"));
    }

    #[test]
    fn add_multiple_subnets_joint5() {
        let mut ip_range = IpRange::new();
        let subnet1 = "172.16.5.130/24".parse().unwrap();
        let subnet2 = "172.16.5.131/16".parse().unwrap();
        ip_range.add(subnet1).add(subnet2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(16, "172.16.0.0"));
    }

    #[test]
    fn add_multiple_subnets_joint6() {
        let mut ip_range = IpRange::new();
        let subnet1 = "172.16.5.130/24".parse().unwrap();
        let subnet2 = "0.0.0.0/0".parse().unwrap();
        ip_range.add(subnet1).add(subnet2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(0, "0.0.0.0"));
    }

    #[test]
    fn remove_subnets_no_split() {
        let mut ip_range = IpRange::new();
        let subnet1 = "192.168.0.0/24".parse().unwrap();
        let subnet2 = "172.16.0.0/16".parse().unwrap();
        ip_range.add(subnet1).add(subnet2).simplify();

        ip_range.remove(subnet1);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(16, "172.16.0.0"));
    }

    #[test]
    fn remove_subnets_split1() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.2.0/23".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            Some("192.168.0.0/23".parse().unwrap()),
            ip_range.get_subnet(23, "192.168.0.0")
        );
    }

    #[test]
    fn remove_subnets_split2() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.0.0/23".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            Some("192.168.2.0/23".parse().unwrap()),
            ip_range.get_subnet(23, "192.168.2.0")
        );
    }

    #[test]
    fn remove_subnets_split3() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.2.0/25".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(
            Some("192.168.0.0/23".parse().unwrap()),
            ip_range.get_subnet(23, "192.168.0.0")
        );
        assert_eq!(
            Some("192.168.2.128/25".parse().unwrap()),
            ip_range.get_subnet(25, "192.168.2.128")
        );
        assert_eq!(
            Some("192.168.3.0/24".parse().unwrap()),
            ip_range.get_subnet(24, "192.168.3.0")
        );
    }


    impl IpRange {
        fn contains_ip(&self, ip: &str) -> bool {
            self.contains(ip.parse::<Ipv4Addr>().unwrap())
        }

        fn find_subnet_by_ip(&self, ip: &str) -> Option<Subnet> {
            self.find_subnet(ip.parse::<Ipv4Addr>().unwrap())
        }

        fn includes_subnet(&self, subnet: &str) -> bool {
            self.includes(subnet.parse().unwrap())
        }

        fn super_subnet_by_subnet(&self, subnet: &str) -> Option<Subnet> {
            self.super_subnet(subnet.parse().unwrap())
        }
    }

    #[test]
    fn contains_ip_with_one_subnet() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/24".parse().unwrap());

        assert!(ip_range.contains_ip("192.168.0.0"));
        assert!(ip_range.contains_ip("192.168.0.128"));
        assert!(ip_range.contains_ip("192.168.0.255"));
        assert!(!ip_range.contains_ip("192.167.255.255"));
        assert!(!ip_range.contains_ip("192.168.1.0"));
    }

    #[test]
    fn contains_ip_with_many_subnets() {
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
    fn find_subnet_with_one_subnet() {
        let mut ip_range = IpRange::new();
        let subnet = "192.168.0.0/24".parse().unwrap();
        ip_range.add(subnet);

        assert_eq!(Some(subnet), ip_range.find_subnet_by_ip("192.168.0.0"));
        assert_eq!(Some(subnet), ip_range.find_subnet_by_ip("192.168.0.128"));
        assert_eq!(Some(subnet), ip_range.find_subnet_by_ip("192.168.0.255"));
        assert_eq!(None, ip_range.find_subnet_by_ip("192.167.255.255"));
        assert_eq!(None, ip_range.find_subnet_by_ip("192.168.1.0"));
    }

    #[test]
    fn find_subnet_with_many_subnets() {
        let mut ip_range = IpRange::new();
        let subnet1 = "192.168.0.0/24".parse().unwrap();
        let subnet2 = "172.16.0.0/16".parse().unwrap();
        let subnet3 = "10.0.0.0/8".parse().unwrap();
        ip_range.add(subnet1).add(subnet2).add(subnet3).simplify();

        assert_eq!(Some(subnet1), ip_range.find_subnet_by_ip("192.168.0.128"));
        assert_eq!(Some(subnet2), ip_range.find_subnet_by_ip("172.16.32.1"));
        assert_eq!(Some(subnet3), ip_range.find_subnet_by_ip("10.10.10.10"));
        assert_eq!(None, ip_range.find_subnet_by_ip("0.0.0.0"));
        assert_eq!(None, ip_range.find_subnet_by_ip("8.8.8.8"));
        assert_eq!(None, ip_range.find_subnet_by_ip("11.0.0.0"));
        assert_eq!(None, ip_range.find_subnet_by_ip("192.167.255.255"));
        assert_eq!(None, ip_range.find_subnet_by_ip("255.255.255.255"));
    }

    #[test]
    fn find_subnet_boundary1() {
        let mut ip_range = IpRange::new();
        let subnet = "0.0.0.0/0".parse().unwrap();
        ip_range.add(subnet);

        assert_eq!(Some(subnet), ip_range.find_subnet_by_ip("0.0.0.0"));
        assert_eq!(Some(subnet), ip_range.find_subnet_by_ip("8.8.8.8"));
        assert_eq!(Some(subnet), ip_range.find_subnet_by_ip("192.168.0.0"));
        assert_eq!(Some(subnet), ip_range.find_subnet_by_ip("192.168.1.1"));
    }

    #[test]
    fn find_subnet_boundary2() {
        let mut ip_range = IpRange::new();
        let subnet = "254.254.254.254/32".parse().unwrap();
        ip_range.add(subnet);

        assert_eq!(None, ip_range.find_subnet_by_ip("0.0.0.0"));
        assert_eq!(None, ip_range.find_subnet_by_ip("8.8.8.8"));
        assert_eq!(None, ip_range.find_subnet_by_ip("192.168.0.0"));
        assert_eq!(Some(subnet), ip_range.find_subnet_by_ip("254.254.254.254"));
    }

    #[test]
    fn includes_subnet_with_one_subnet() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/24".parse().unwrap());

        assert!(ip_range.includes_subnet("192.168.0.0/24"));
        assert!(ip_range.includes_subnet("192.168.0.128/25"));
        assert!(!ip_range.includes_subnet("192.168.0.0/23"));
        assert!(!ip_range.includes_subnet("192.168.1.0/24"));
        assert!(!ip_range.includes_subnet("192.167.0.0/24"));
    }

    #[test]
    fn includes_subnet_with_many_subnets() {
        let mut ip_range = IpRange::new();
        ip_range
            .add("192.168.0.0/24".parse().unwrap())
            .add("172.16.0.0/16".parse().unwrap())
            .add("10.0.0.0/8".parse().unwrap())
            .simplify();

        assert!(ip_range.includes_subnet("192.168.0.128/25"));
        assert!(ip_range.includes_subnet("172.16.32.0/20"));
        assert!(ip_range.includes_subnet("10.10.0.0/16"));
        assert!(!ip_range.includes_subnet("0.0.0.0/0"));
        assert!(!ip_range.includes_subnet("8.0.0.0/6"));
        assert!(!ip_range.includes_subnet("8.0.0.0/7"));
        assert!(!ip_range.includes_subnet("11.0.0.0/9"));
        assert!(!ip_range.includes_subnet("192.167.255.255/32"));
        assert!(!ip_range.includes_subnet("255.0.0.0/8"));
    }

    #[test]
    fn includes_subnet_boundary1() {
        let mut ip_range = IpRange::new();
        ip_range.add("0.0.0.0/0".parse().unwrap());

        assert!(ip_range.includes_subnet("0.0.0.0/0"));
        assert!(ip_range.includes_subnet("8.0.0.0/6"));
        assert!(ip_range.includes_subnet("11.0.0.0/9"));
        assert!(ip_range.includes_subnet("192.168.0.128/25"));
        assert!(ip_range.includes_subnet("255.255.255.255/32"));
    }

    #[test]
    fn includes_subnet_boundary2() {
        let mut ip_range = IpRange::new();
        ip_range.add("254.254.254.254/32".parse().unwrap());

        assert!(!ip_range.includes_subnet("0.0.0.0/0"));
        assert!(!ip_range.includes_subnet("8.0.0.0/6"));
        assert!(!ip_range.includes_subnet("254.254.0.0/16"));
        assert!(ip_range.includes_subnet("254.254.254.254/32"));
        assert!(!ip_range.includes_subnet("255.255.255.255/32"));
    }

    #[test]
    fn super_subnet_with_one_subnet() {
        let mut ip_range = IpRange::new();
        let subnet = "192.168.0.0/24".parse().unwrap();
        ip_range.add(subnet);

        assert_eq!(
            Some(subnet),
            ip_range.super_subnet_by_subnet("192.168.0.0/24")
        );
        assert_eq!(
            Some(subnet),
            ip_range.super_subnet_by_subnet("192.168.0.128/25")
        );
        assert_eq!(None, ip_range.super_subnet_by_subnet("192.168.0.0/23"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("192.168.1.0/24"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("192.167.0.0/24"));
    }

    #[test]
    fn super_subnet_with_many_subnets() {
        let mut ip_range = IpRange::new();
        let subnet1 = "192.168.0.0/24".parse().unwrap();
        let subnet2 = "172.16.0.0/16".parse().unwrap();
        let subnet3 = "10.0.0.0/8".parse().unwrap();
        ip_range.add(subnet1).add(subnet2).add(subnet3).simplify();

        assert_eq!(
            Some(subnet1),
            ip_range.super_subnet_by_subnet("192.168.0.128/25")
        );
        assert_eq!(
            Some(subnet2),
            ip_range.super_subnet_by_subnet("172.16.32.0/20")
        );
        assert_eq!(
            Some(subnet3),
            ip_range.super_subnet_by_subnet("10.10.0.0/16")
        );
        assert_eq!(None, ip_range.super_subnet_by_subnet("0.0.0.0/0"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("8.0.0.0/6"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("8.0.0.0/7"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("11.0.0.0/9"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("192.167.255.255/32"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("255.0.0.0/8"));
    }

    #[test]
    fn super_subnet_boundary1() {
        let mut ip_range = IpRange::new();
        let subnet = "0.0.0.0/0".parse().unwrap();
        ip_range.add(subnet);

        assert_eq!(Some(subnet), ip_range.super_subnet_by_subnet("0.0.0.0/0"));
        assert_eq!(Some(subnet), ip_range.super_subnet_by_subnet("8.0.0.0/6"));
        assert_eq!(Some(subnet), ip_range.super_subnet_by_subnet("11.0.0.0/9"));
        assert_eq!(
            Some(subnet),
            ip_range.super_subnet_by_subnet("192.168.0.128/25")
        );
        assert_eq!(
            Some(subnet),
            ip_range.super_subnet_by_subnet("255.255.255.255/32")
        );
    }

    #[test]
    fn super_subnet_boundary2() {
        let mut ip_range = IpRange::new();
        let subnet = "254.254.254.254/32".parse().unwrap();
        ip_range.add(subnet);

        assert_eq!(None, ip_range.super_subnet_by_subnet("0.0.0.0/0"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("8.0.0.0/6"));
        assert_eq!(None, ip_range.super_subnet_by_subnet("254.254.0.0/16"));
        assert_eq!(
            Some(subnet),
            ip_range.super_subnet_by_subnet("254.254.254.254/32")
        );
        assert_eq!(None, ip_range.super_subnet_by_subnet("255.255.255.255/32"));
    }

    #[test]
    fn merge_empty1() {
        let ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "10.0.0.1/8".parse().unwrap();
        let subnet2 = "172.16.5.130/16".parse().unwrap();
        let subnet3 = "192.168.1.1/24".parse().unwrap();
        let subnet4 = "254.254.254.254/32".parse().unwrap();
        ip_range2
            .add(subnet1)
            .add(subnet2)
            .add(subnet3)
            .add(subnet4)
            .simplify();

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(subnet1), ip_range.get_subnet(8, "10.0.0.0"));
        assert_eq!(Some(subnet2), ip_range.get_subnet(16, "172.16.0.0"));
        assert_eq!(Some(subnet3), ip_range.get_subnet(24, "192.168.1.0"));
        assert_eq!(Some(subnet4), ip_range.get_subnet(32, "254.254.254.254"));
    }

    #[test]
    fn merge_empty2() {
        let mut ip_range1 = IpRange::new();
        let ip_range2 = IpRange::new();
        let subnet1 = "10.0.0.1/8".parse().unwrap();
        let subnet2 = "172.16.5.130/16".parse().unwrap();
        let subnet3 = "192.168.1.1/24".parse().unwrap();
        let subnet4 = "254.254.254.254/32".parse().unwrap();
        ip_range1
            .add(subnet1)
            .add(subnet2)
            .add(subnet3)
            .add(subnet4)
            .simplify();

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(subnet1), ip_range.get_subnet(8, "10.0.0.0"));
        assert_eq!(Some(subnet2), ip_range.get_subnet(16, "172.16.0.0"));
        assert_eq!(Some(subnet3), ip_range.get_subnet(24, "192.168.1.0"));
        assert_eq!(Some(subnet4), ip_range.get_subnet(32, "254.254.254.254"));
    }

    #[test]
    fn merge_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "10.0.0.1/8".parse().unwrap();
        let subnet2 = "172.16.5.130/16".parse().unwrap();
        let subnet3 = "192.168.1.1/24".parse().unwrap();
        let subnet4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(subnet1).add(subnet2);
        ip_range2.add(subnet3).add(subnet4);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(subnet1), ip_range.get_subnet(8, "10.0.0.0"));
        assert_eq!(Some(subnet2), ip_range.get_subnet(16, "172.16.0.0"));
        assert_eq!(Some(subnet3), ip_range.get_subnet(24, "192.168.1.0"));
        assert_eq!(Some(subnet4), ip_range.get_subnet(32, "254.254.254.254"));
    }

    #[test]
    fn merge_joint1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.4.130/24".parse().unwrap();
        let subnet2 = "172.16.5.130/22".parse().unwrap();
        ip_range1.add(subnet1);
        ip_range2.add(subnet2);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(22, "172.16.4.0"));
    }

    #[test]
    fn merge_joint2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.5.130/24".parse().unwrap();
        let subnet2 = "172.16.4.130/22".parse().unwrap();
        ip_range1.add(subnet1);
        ip_range2.add(subnet2);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(22, "172.16.4.0"));
    }

    #[test]
    fn merge_sequent1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.4.0/24".parse().unwrap();
        let subnet2 = "172.16.5.0/24".parse().unwrap();
        let subnet3 = "172.16.6.0/24".parse().unwrap();
        ip_range1.add(subnet1);
        ip_range2.add(subnet2);
        ip_range2.add(subnet3);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 2);
        assert_eq!(
            "172.16.4.0/23".parse().ok(),
            ip_range.get_subnet(23, "172.16.4.0")
        );
        assert_eq!(
            "172.16.6.0/24".parse().ok(),
            ip_range.get_subnet(24, "172.16.6.0")
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

        let ip_range = ip_range1.merge(&ip_range2).merge(&ip_range3);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            "192.168.0.0/19".parse().ok(),
            ip_range.get_subnet(19, "192.168.0.0")
        );
    }

    #[test]
    fn intersect_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "10.0.0.1/8".parse().unwrap();
        let subnet2 = "172.16.5.130/16".parse().unwrap();
        let subnet3 = "192.168.1.1/24".parse().unwrap();
        let subnet4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(subnet1).add(subnet2);
        ip_range2.add(subnet3).add(subnet4);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 0);
    }

    #[test]
    fn intersect_joint1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.4.130/24".parse().unwrap();
        let subnet2 = "172.16.5.130/22".parse().unwrap();
        ip_range1.add(subnet1);
        ip_range2.add(subnet2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet1), ip_range.get_subnet(24, "172.16.4.0"));
    }

    #[test]
    fn intersect_joint2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.5.130/24".parse().unwrap();
        let subnet2 = "172.16.4.130/22".parse().unwrap();
        ip_range1.add(subnet1);
        ip_range2.add(subnet2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet1), ip_range.get_subnet(24, "172.16.5.0"));
    }

    #[test]
    fn intersect_joint3() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.5.130/24".parse().unwrap();
        let subnet2 = "172.16.5.130/24".parse().unwrap();
        ip_range1.add(subnet1);
        ip_range2.add(subnet2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet1), ip_range.get_subnet(24, "172.16.5.0"));
    }

    #[test]
    fn intersect_joint4() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "10.0.0.0/8".parse().unwrap();
        let subnet2 = "192.168.0.0/24".parse().unwrap();
        let subnet3 = "10.10.0.0/16".parse().unwrap();
        let subnet4 = "10.254.0.0/17".parse().unwrap();
        let subnet5 = "192.168.0.0/16".parse().unwrap();
        ip_range1.add(subnet1).add(subnet2);
        ip_range2.add(subnet3).add(subnet4).add(subnet5);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(subnet3), ip_range.get_subnet(16, "10.10.0.0"));
        assert_eq!(Some(subnet4), ip_range.get_subnet(17, "10.254.0.0"));
        assert_eq!(Some(subnet2), ip_range.get_subnet(24, "192.168.0.0"));
    }

    #[test]
    fn exclude_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "10.0.0.1/8".parse().unwrap();
        let subnet2 = "172.16.5.130/16".parse().unwrap();
        let subnet3 = "192.168.1.1/24".parse().unwrap();
        let subnet4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(subnet1).add(subnet2);
        ip_range2.add(subnet3).add(subnet4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range1, ip_range);
    }

    #[test]
    fn exclude_larger() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.4.130/24".parse().unwrap();
        let subnet2 = "192.168.1.0/24".parse().unwrap();
        let subnet3 = "172.16.5.130/22".parse().unwrap();
        ip_range1.add(subnet1).add(subnet2);
        ip_range2.add(subnet3);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(24, "192.168.1.0"));
    }

    #[test]
    fn exclude_identical() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.5.130/24".parse().unwrap();
        let subnet2 = "192.168.1.0/24".parse().unwrap();
        let subnet3 = "172.16.5.130/22".parse().unwrap();
        let subnet4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(subnet1).add(subnet2);
        ip_range2.add(subnet3).add(subnet4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(subnet2), ip_range.get_subnet(24, "192.168.1.0"));
    }

    #[test]
    fn exclude_split1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.5.0/22".parse().unwrap();
        let subnet2 = "192.168.1.0/24".parse().unwrap();
        let subnet3 = "172.16.5.130/24".parse().unwrap();
        let subnet4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(subnet1).add(subnet2);
        ip_range2.add(subnet3).add(subnet4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(subnet2), ip_range.get_subnet(24, "192.168.1.0"));
        assert_eq!(
            "172.16.4.0/24".parse().ok(),
            ip_range.get_subnet(24, "172.16.4.0")
        );
        assert_eq!(
            "172.16.6.0/23".parse().ok(),
            ip_range.get_subnet(23, "172.16.6.0")
        );
    }

    #[test]
    fn exclude_split2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let subnet1 = "172.16.4.0/22".parse().unwrap();
        let subnet2 = "192.168.1.0/24".parse().unwrap();
        let subnet3 = "172.16.4.0/24".parse().unwrap();
        let subnet4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(subnet1).add(subnet2);
        ip_range2.add(subnet3).add(subnet4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(subnet2), ip_range.get_subnet(24, "192.168.1.0"));
        assert_eq!(
            "172.16.5.0/24".parse().ok(),
            ip_range.get_subnet(24, "172.16.5.0")
        );
        assert_eq!(
            "172.16.6.0/23".parse().ok(),
            ip_range.get_subnet(23, "172.16.6.0")
        );
    }
}
