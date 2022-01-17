# iprange-rs

[![Crates Version](https://img.shields.io/crates/v/iprange.svg)](https://crates.io/crates/iprange)
[![Build Status](https://travis-ci.org/sticnarf/iprange-rs.svg?branch=master)](https://travis-ci.org/sticnarf/iprange-rs)

`iprange-rs` is a Rust library for managing IP ranges. 

It provides fast adding and removing operations.

It also provides `merge`, `intersect` and `exclude` methods 
that enable you to manipulate it like a set.

Of course, you can test whether an IP address is in an `IpRange`.

**See the [documentation](https://docs.rs/iprange/) for details.**

## Example

```rust
extern crate iprange;
extern crate ipnet;

use std::net::Ipv4Addr;
use iprange::IpRange;
use ipnet::Ipv4Net;

fn main() {
    let ip_range: IpRange<Ipv4Net> = ["10.0.0.0/8", "172.16.0.0/16", "192.168.1.0/24"]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();

    assert!(ip_range.contains(&"172.16.32.1".parse::<Ipv4Addr>().unwrap()));
    assert!(ip_range.contains(&"192.168.1.1".parse::<Ipv4Addr>().unwrap()));
}
```

## Serde support

Serde support is optional and disabled by default. To enable, use the feature `serde`.

```toml
[dependencies]
iprange = { version = "0.6", features = ["serde"] }
```

## Benchmark

`iprange-rs` stores the IP networks in a radix trie.
This allows us to store and lookup IP information quickly.

There is no Rust alternative to this crate, so I decide to compare it to those written in Go.

On my computer, here is the [benchmark](https://github.com/smallnest/iprange) result for Go implementations: 

```
BenchmarkIPv4Contains-8                   500000              2545 ns/op
BenchmarkIPv4Contains_Radix-8             200000              6960 ns/op
BenchmarkIPv4Contains_NRadix-8           1000000              1828 ns/op
BenchmarkIPv6Contains-8                   300000              3989 ns/op
BenchmarkIPv6Contains_Radix-8             200000              6818 ns/op
BenchmarkIPv6Contains_NRadix-8            500000              3039 ns/op
```

And below are the results of the equivalent Rust program using `iprange-rs`:

```
test test_ipv4_against_go             ... bench:         751 ns/iter (+/- 5)
test test_ipv6_against_go             ... bench:       2,500 ns/iter (+/- 20)
```

We can see the Rust one using `iprange-rs` is **2.4x faster** than
even the fastest Go implementation when dealing with IPv4 and is 1.2x faster with IPv6.

## License

`iprange-rs` is licensed under the MIT license.
