#![feature(test)]

extern crate ipnet;
extern crate iprange;
extern crate rand;
extern crate test;

use test::Bencher;
use iprange::*;
use ipnet::{Ipv4Net, Ipv6Net};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::fs::File;
use std::path::PathBuf;
use std::io::{BufRead, BufReader};
use rand::{Rng, SeedableRng, StdRng};

#[bench]
fn parse_one_ipv4_addr(b: &mut Bencher) {
    b.iter(|| "192.168.1.1".parse::<Ipv4Addr>().unwrap());
}

#[bench]
fn parse_one_ipv4_net(b: &mut Bencher) {
    b.iter(|| "192.168.0.0/24".parse::<Ipv4Net>().unwrap());
}

#[bench]
fn parse_one_ipv6_addr(b: &mut Bencher) {
    b.iter(|| "2620:0:ccc::2".parse::<Ipv6Addr>().unwrap());
}

#[bench]
fn parse_one_ipv6_net(b: &mut Bencher) {
    b.iter(|| "2400:9dc0::/32".parse::<Ipv6Net>().unwrap());
}


fn read_lines_from_file(file_name: &str) -> Vec<String> {
    let path = PathBuf::from(file!());
    let f = File::open(path.with_file_name(file_name)).expect("Unable to open file");
    let reader = BufReader::new(f);
    reader.lines().flat_map(|l| l).collect()
}

fn chnlists_v4() -> Vec<String> {
    read_lines_from_file("chnlist.txt")
}

fn chnlists_v6() -> Vec<String> {
    read_lines_from_file("chnlist-v6.txt")
}

fn rand_ipv4_list(n: usize) -> Vec<Ipv4Addr> {
    let mut rng = StdRng::from_seed(&[1926, 8, 17]);
    (0..n).map(|_| rng.next_u32().into()).collect()
}

fn rand_ipv6_list(n: usize) -> Vec<Ipv6Addr> {
    let mut rng = StdRng::from_seed(&[1926, 8, 17]);
    (0..n)
        .map(|_| {
            let mut buf = [0u8; 16];
            rng.fill_bytes(&mut buf);
            buf.into()
        })
        .collect()
}

#[bench]
fn parse_chnlists_v4(b: &mut Bencher) {
    let lines = chnlists_v4();
    b.iter(|| for line in &lines {
        line.parse::<Ipv4Net>().ok();
    });
}

#[bench]
fn create_ip_range_with_chnlists_v4(b: &mut Bencher) {
    let chnlists = chnlists_v4();
    b.iter(|| {
        chnlists
            .iter()
            .flat_map(|l| l.parse::<Ipv4Net>())
            .collect::<IpRange<Ipv4Net>>()
    });
}

#[bench]
fn test_10000_ips_in_chnlists_v4(b: &mut Bencher) {
    let ip_list = rand_ipv4_list(10000);
    let chnlists = chnlists_v4()
        .iter()
        .flat_map(|l| l.parse::<Ipv4Net>())
        .collect::<IpRange<Ipv4Net>>();
    b.iter(|| for ip in &ip_list {
        chnlists.contains(ip);
    });
}

#[bench]
fn parse_chnlists_v6(b: &mut Bencher) {
    let lines = chnlists_v6();
    b.iter(|| for line in &lines {
        line.parse::<Ipv6Net>().ok();
    });
}

#[bench]
fn create_ip_range_with_chnlists_v6(b: &mut Bencher) {
    let chnlists = chnlists_v6();
    b.iter(|| {
        chnlists
            .iter()
            .flat_map(|l| l.parse::<Ipv6Net>())
            .collect::<IpRange<Ipv6Net>>()
    });
}

#[bench]
fn test_10000_ips_in_chnlists_v6(b: &mut Bencher) {
    let ip_list = rand_ipv6_list(10000);
    let chnlists = chnlists_v6()
        .iter()
        .flat_map(|l| l.parse::<Ipv6Net>())
        .collect::<IpRange<Ipv6Net>>();
    b.iter(|| for ip in &ip_list {
        chnlists.contains(ip);
    });
}

// #[bench]
// fn test_ipv4_against_go(b: &mut Bencher) {
//     let ip_range = read_lines_from_file("cidr_ipv4_test.data")
//         .iter()
//         .flat_map(|l| l.parse::<Ipv4Net>())
//         .collect::<IpRange<Ipv4Net>>();
//     b.iter(|| {
//         assert!(ip_range.contains(&"103.67.32.0".parse::<Ipv4Addr>().unwrap()));
//         assert!(ip_range.contains(&"103.67.32.1".parse::<Ipv4Addr>().unwrap()));
//         assert!(!ip_range
//             .contains(&"103.67.100.77".parse::<Ipv4Addr>().unwrap()));
//         assert!(ip_range.contains(&"3.0.0.0".parse::<Ipv4Addr>().unwrap()));
//         assert!(ip_range.contains(&"216.255.255.255".parse::<Ipv4Addr>().unwrap()));
//         assert!(!ip_range
//             .contains(&"2.255.255.255".parse::<Ipv4Addr>().unwrap()));
//         assert!(!ip_range
//             .contains(&"217.0.0.0".parse::<Ipv4Addr>().unwrap()));
//         assert!(!ip_range.contains(&"0.0.0.0".parse::<Ipv4Addr>().unwrap()));
//         assert!(!ip_range.contains(&"255.255.255.255".parse::<Ipv4Addr>().unwrap()));
//     });
// }

// #[bench]
// fn test_ipv6_against_go(b: &mut Bencher) {
//     let ip_range = read_lines_from_file("cidr_ipv6_test.data")
//         .iter()
//         .flat_map(|l| l.parse::<Ipv6Net>())
//         .collect::<IpRange<Ipv6Net>>();
//     b.iter(|| {
//         assert!(ip_range.contains(&"2607:d200::".parse::<Ipv6Addr>().unwrap()));
//         assert!(ip_range.contains(&"2607:d200::1".parse::<Ipv6Addr>().unwrap()));
//         assert!(!ip_range.contains(&"2607:d201::ffff".parse::<Ipv6Addr>().unwrap()));
//         assert!(ip_range.contains(&"2001:1800::".parse::<Ipv6Addr>().unwrap()));
//         assert!(
//             ip_range.contains(&"2a03:cd00:ffff:ffff:ffff:ffff:ffff:ffff"
//                 .parse::<Ipv6Addr>()
//                 .unwrap())
//         );
//         assert!(!ip_range.contains(
//             &"2001:17ff:ffff:ffff:ffff:ffff:ffff:ffff"
//                 .parse::<Ipv6Addr>()
//                 .unwrap()
//         ));
//         assert!(!ip_range
//             .contains(&"2a03:cd01::".parse::<Ipv6Addr>().unwrap()));
//         assert!(!ip_range.contains(&"::".parse::<Ipv6Addr>().unwrap()));
//         assert!(!ip_range.contains(
//             &"ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"
//                 .parse::<Ipv6Addr>()
//                 .unwrap()
//         ));
//     });
// }
