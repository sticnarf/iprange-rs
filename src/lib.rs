extern crate byteorder;

use std::str::FromStr;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::net::Ipv4Addr;
use std::collections::{btree_map, BTreeMap, Bound};
use byteorder::{BigEndian, ByteOrder};

pub struct IpRange {
    subnets: BTreeMap<CompactIpv4, Subnet>, // Key: prefix
}

impl IpRange {
    pub fn new() -> IpRange {
        IpRange {
            subnets: BTreeMap::new(),
        }
    }

    pub fn push(&mut self, subnet: Subnet) {
        let prev = self.prev_subnet(subnet);
        let link_range = self.next_subnet(subnet)
            .map_or(Bound::Unbounded, |next_subnet| {
                Bound::Excluded(next_subnet.prefix)
            });
        let link = self.subnets
            .range((Bound::Included(subnet.prefix), link_range))
            .next_back()
            .map(|(_, &subnet)| subnet);

        let prev_inclusive = prev.map_or(false, |prev| prev.contains(subnet.prefix));
        let link_inclusive = link.map_or(false, |link| link.contains(subnet.end()));

        let mut new_subnet = subnet;
        if prev_inclusive {
            let prev_subnet = prev.unwrap();
            self.subnets.remove(&prev_subnet.prefix);
            new_subnet.prefix = prev_subnet.prefix;
            new_subnet.mask &= prev_subnet.mask;
        }
        if link_inclusive {
            let link_subnet = link.unwrap();
            self.subnets.remove(&link_subnet.prefix);
            new_subnet.mask &= link_subnet.mask;
        }

        let range = (
            prev.map_or(Bound::Unbounded, |prev| Bound::Excluded(prev.prefix)),
            Bound::Excluded(subnet.end()),
        );

        let to_be_deleted: Vec<CompactIpv4> = self.subnets
            .range(range)
            .map(|(&prefix, _)| prefix)
            .collect();

        for prefix in to_be_deleted {
            self.subnets.remove(&prefix);
        }

        self.subnets.insert(new_subnet.prefix, new_subnet);
    }

    pub fn contains(&self, addr: CompactIpv4) -> bool {
        self.candidate(addr)
            .map_or(false, |subnet| subnet.contains(addr))
    }

    fn prev_subnet(&self, subnet: Subnet) -> Option<Subnet> {
        self.candidate(subnet.prefix)
    }

    fn next_subnet(&self, subnet: Subnet) -> Option<Subnet> {
        self.subnets
            .range((Bound::Excluded(subnet.end()), Bound::Unbounded))
            .next()
            .map(|(_, &subnet)| subnet)
    }

    fn candidate(&self, addr: CompactIpv4) -> Option<Subnet> {
        self.subnets
            .range((Bound::Unbounded, Bound::Included(addr)))
            .next_back()
            .map(|(_, &subnet)| subnet)
    }
}

impl<'a> IntoIterator for &'a IpRange {
    type Item = Subnet;
    type IntoIter = IpRangeIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IpRangeIter {
            map_iter: self.subnets.iter(),
        }
    }
}

pub struct IpRangeIter<'a> {
    map_iter: btree_map::Iter<'a, CompactIpv4, Subnet>,
}

impl<'a> Iterator for IpRangeIter<'a> {
    type Item = Subnet;

    fn next(&mut self) -> Option<Subnet> {
        self.map_iter.next().map(|(_, &subnet)| subnet)
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

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Subnet {
    pub prefix: CompactIpv4,
    pub mask: CompactIpv4,
}

impl Subnet {
    pub fn contains(&self, addr: CompactIpv4) -> bool {
        (addr & self.mask) == self.prefix
    }

    pub fn end(&self) -> CompactIpv4 {
        self.prefix | !(self.mask)
    }
}

impl fmt::Debug for Subnet {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mask: u32 = self.mask.into();
        write!(f, "{:?}/{}", self.prefix, 32 - mask.trailing_zeros())
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
                            mask,
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
    fn parse_subnet() {
        let my_subnet: Subnet = "192.168.5.130/24".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11000000_10101000_00000101_00000000.into(),
                mask: 0b11111111_11111111_11111111_00000000.into(),
            }
        );

        let my_subnet: Subnet = "192.168.5.130/255.255.255.192".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11000000_10101000_00000101_10000000.into(),
                mask: 0b11111111_11111111_11111111_11000000.into(),
            }
        );

        let my_subnet: Subnet = "10.20.30.40/8".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b00001010_00000000_00000000_00000000.into(),
                mask: 0b11111111_00000000_00000000_00000000.into(),
            }
        );

        let my_subnet: Subnet = "233.233.233.233/32".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11101001_11101001_11101001_11101001.into(),
                mask: 0b11111111_11111111_11111111_11111111.into(),
            }
        );
    }

    #[test]
    fn with_single_subnet() {
        let mut ip_range = IpRange::new();
        ip_range.push("192.168.5.130/24".parse().unwrap());
        let ip: Ipv4Addr = "192.168.5.1".parse().unwrap();
        assert!(ip_range.contains(ip.into()));
        let ip: Ipv4Addr = "192.168.6.1".parse().unwrap();
        assert!(!ip_range.contains(ip.into()));
    }

    #[test]
    fn with_multiple_subnets() {
        let mut ip_range = IpRange::new();
        ip_range.push("172.16.5.130/24".parse().unwrap());
        let ip: Ipv4Addr = "172.16.6.1".parse().unwrap();
        assert!(!ip_range.contains(ip.into()));
        ip_range.push("172.16.3.1/16".parse().unwrap());
        assert!(ip_range.contains(ip.into()));
        let ip: Ipv4Addr = "172.16.31.1".parse().unwrap();
        assert!(ip_range.contains(ip.into()));
        let ip: Ipv4Addr = "172.17.0.1".parse().unwrap();
        assert!(!ip_range.contains(ip.into()));

        let mut ip_range = IpRange::new();
        ip_range.push("192.168.5.1/24".parse().unwrap());
        ip_range.push("192.168.6.1/22".parse().unwrap());
        ip_range.push("192.168.128.1/20".parse().unwrap());
        let ip: Ipv4Addr = "192.168.1.1".parse().unwrap();
        assert!(!ip_range.contains(ip.into()));
        let ip: Ipv4Addr = "192.168.4.1".parse().unwrap();
        assert!(ip_range.contains(ip.into()));
        let ip: Ipv4Addr = "192.168.130.1".parse().unwrap();
        assert!(ip_range.contains(ip.into()));
    }

    #[test]
    fn iter_ip_range() {
        let mut ip_range = IpRange::new();
        ip_range.push("192.168.5.1/24".parse().unwrap());
        ip_range.push("192.168.6.1/22".parse().unwrap());
        ip_range.push("192.168.128.1/20".parse().unwrap());

        let ranges: Vec<String> = ip_range
            .into_iter()
            .map(|subnet| format!("{:?}", subnet))
            .collect();
        assert_eq!(ranges, vec!["192.168.4.0/22", "192.168.128.0/20"]);
    }
}
