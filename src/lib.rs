extern crate byteorder;

use std::str::FromStr;
use std::sync::Arc;
use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::net::Ipv4Addr;
use std::iter::FromIterator;
use std::collections::{btree_map, BTreeMap, Bound};
use byteorder::{BigEndian, ByteOrder};

#[derive(Clone)]
pub struct IpRange {
    // The index of the Vec is the prefix size of the subnets
    // which the corresponding BTreeMap contains
    subnets: Vec<BTreeMap<CompactIpv4, Subnet>>,
}

impl IpRange {
    pub fn new() -> IpRange {
        IpRange {
            subnets: (0..33).map(|_| BTreeMap::new()).collect(),
        }
    }

    pub fn add(&mut self, subnet: Subnet) -> &mut IpRange {
        if !self.includes(subnet) {
            self.remove_inside(subnet);
            self.subnets[subnet.prefix_size].insert(subnet.prefix, subnet);
        }
        self
    }

    pub fn remove(&mut self, subnet: Subnet) {
        self.remove_inside(subnet);

        while let Some(super_subnet) = self.super_subnet(subnet) {
            self.split_subnet(super_subnet);
            self.remove_subnet(subnet);
        }
    }

    pub fn merge(&self, other: &IpRange) -> IpRange {
        unimplemented!()
        // self.into_iter().chain(other.into_iter()).collect()
    }

    pub fn intersect(&self, other: &IpRange) -> IpRange {
        unimplemented!()
        // let range1 = self.into_iter().filter(|&subnet| other.includes(subnet));
        // let range2 = other.into_iter().filter(|&subnet| self.includes(subnet));
        // range1.chain(range2).collect()
    }

    pub fn exclude(&self, other: &IpRange) -> IpRange {
        unimplemented!()
        // let mut new = self.clone();
        // for subnet in other.into_iter() {
        //     new.remove(subnet);
        // }
        // new
    }

    pub fn contains<T: Into<CompactIpv4>>(&self, addr: T) -> bool {
        self.find_subnet(addr.into()).is_some()
    }

    pub fn includes(&self, subnet: Subnet) -> bool {
        self.super_subnet(subnet).is_some()
    }

    fn super_subnet(&self, subnet: Subnet) -> Option<Subnet> {
        for larger in self.subnets[0..(subnet.prefix_size + 1)].iter() {
            if let Some((&larger_prefix, &larger_subnet)) = larger
                .range((Bound::Unbounded, Bound::Included(subnet.prefix)))
                .next_back()
            {
                if subnet.prefix & larger_subnet.mask() == larger_prefix {
                    return Some(larger_subnet);
                }
            }
        }
        None
    }

    fn find_subnet(&self, ip: CompactIpv4) -> Option<Subnet> {
        self.super_subnet(ip.into())
    }

    fn split_subnet(&mut self, subnet: Subnet) {
        assert!(subnet.prefix_size < 32);

        self.remove_subnet(subnet);
        self.add_subnet(subnet.prefix, subnet.prefix_size + 1);
        self.add_subnet(
            (subnet.prefix.0 | (0xffffffffu32 >> (subnet.prefix_size + 1)) + 1).into(),
            subnet.prefix_size + 1,
        )
    }

    fn remove_inside(&mut self, subnet: Subnet) {
        for smaller in self.subnets[subnet.prefix_size..33].iter_mut() {
            let to_be_removed: Vec<CompactIpv4> = smaller
                .range((
                    Bound::Included(subnet.prefix),
                    Bound::Included(subnet.end()),
                ))
                .map(|(&prefix, _)| prefix)
                .collect();
            for prefix in to_be_removed {
                smaller.remove(&prefix);
            }
        }
    }

    fn add_subnet(&mut self, prefix: CompactIpv4, prefix_size: usize) {
        self.subnets[prefix_size].insert(
            prefix,
            Subnet {
                prefix,
                prefix_size,
            },
        );
    }

    fn remove_subnet(&mut self, subnet: Subnet) {
        self.subnets[subnet.prefix_size].remove(&subnet.prefix);
    }
}

impl<'a> IntoIterator for &'a IpRange {
    type Item = &'a Subnet;
    type IntoIter = IpRangeIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IpRangeIter {
            iter: Box::new(
                self.subnets
                    .iter()
                    .flat_map(|m| m.iter().map(|(_, subnet)| subnet)),
            ),
        }
    }
}

pub struct IpRangeIter<'a> {
    iter: Box<Iterator<Item = &'a Subnet> + 'a>,
}

impl<'a> Iterator for IpRangeIter<'a> {
    type Item = &'a Subnet;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize {
        self.iter.count()
    }
}

impl<'a> FromIterator<Subnet> for IpRange {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Subnet>,
    {
        let mut ip_range = IpRange::new();
        iter.into_iter().fold((), |_, subnet| {
            ip_range.add(subnet);
        });
        ip_range
    }
}

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
    pub fn successor(self) -> Option<CompactIpv4> {
        self.0.checked_add(1).map(|v| CompactIpv4(v))
    }

    pub fn predecessor(self) -> Option<CompactIpv4> {
        self.0.checked_sub(1).map(|v| CompactIpv4(v))
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Subnet {
    pub prefix: CompactIpv4,
    pub prefix_size: usize,
}

impl Subnet {
    pub fn contains<T: Into<CompactIpv4>>(&self, addr: T) -> bool {
        (addr.into() & self.mask()) == self.prefix
    }

    pub fn end(&self) -> CompactIpv4 {
        self.prefix | !(self.mask())
    }

    pub fn includes(&self, other: Subnet) -> bool {
        self.prefix <= other.prefix && self.end() >= other.end()
    }

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

    impl IpRange {
        fn get_subnet(&self, prefix_size: usize, prefix: &str) -> Option<Subnet> {
            self.subnets[prefix_size]
                .get(&prefix.parse::<Ipv4Addr>().unwrap().into())
                .map(|&subnet| subnet)
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
        ip_range.add(subnet1).add(subnet2).add(subnet3).add(subnet4);

        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(subnet1), ip_range.get_subnet(8, "10.0.0.0"));
        assert_eq!(Some(subnet2), ip_range.get_subnet(16, "172.16.0.0"));
        assert_eq!(Some(subnet3), ip_range.get_subnet(24, "192.168.1.0"));
        assert_eq!(Some(subnet4), ip_range.get_subnet(32, "254.254.254.254"));
    }

    #[test]
    fn add_multiple_subnets_joint1() {
        let mut ip_range = IpRange::new();
        let subnet1 = "172.16.4.130/24".parse().unwrap();
        let subnet2 = "172.16.5.130/22".parse().unwrap();
        ip_range.add(subnet1).add(subnet2);

        assert_eq!(Some(subnet2), ip_range.get_subnet(22, "172.16.4.0"));
        assert_eq!(None, ip_range.get_subnet(24, "172.16.4.0"));
    }

    #[test]
    fn with_multiple_subnets() {
        let mut ip_range = IpRange::new();
        ip_range.add("172.16.5.130/24".parse().unwrap());
        let ip: Ipv4Addr = "172.16.6.1".parse().unwrap();
        assert!(!ip_range.contains(ip));
        ip_range.add("172.16.3.1/16".parse().unwrap());
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "172.16.31.1".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "172.17.0.1".parse().unwrap();
        assert!(!ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.5.1/24".parse().unwrap());
        ip_range.add("192.168.6.1/22".parse().unwrap());
        ip_range.add("192.168.128.1/20".parse().unwrap());
        let ip: Ipv4Addr = "192.168.1.1".parse().unwrap();
        assert!(!ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.4.1".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.130.1".parse().unwrap();
        assert!(ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/24".parse().unwrap());
        ip_range.add("192.168.1.0/24".parse().unwrap());
        let ip: Ipv4Addr = "192.168.0.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.1.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.2.252".parse().unwrap();
        assert!(!ip_range.contains(ip));
        ip_range.add("192.168.0.0/18".parse().unwrap());
        assert!(ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range.add("192.168.4.0/24".parse().unwrap());
        let ip: Ipv4Addr = "192.168.0.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.2.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.4.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.5.252".parse().unwrap();
        assert!(!ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.6.252".parse().unwrap();
        assert!(!ip_range.contains(ip));
    }

    #[test]
    fn remove_subnets() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range.remove("192.168.2.0/23".parse().unwrap());
        let ip: Ipv4Addr = "192.168.0.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.1.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.2.252".parse().unwrap();
        assert!(!ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range.remove("192.168.0.0/23".parse().unwrap());
        let ip: Ipv4Addr = "192.168.2.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.3.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.1.252".parse().unwrap();
        assert!(!ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/16".parse().unwrap());
        ip_range.remove("192.168.128.0/24".parse().unwrap());
        let ip: Ipv4Addr = "192.168.127.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.1.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.129.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.252.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.128.252".parse().unwrap();
        assert!(!ip_range.contains(ip));
    }
}
