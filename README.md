# Spanning tree representation for evolutionary algorithms

This project contains some implementations of spanning trees data structures
that can be used in evolutionary algorithms.

## Compiling and executing the experiments

To compile this project it is necessary to have Rust
[installed][install-rust].

To compile the project, execute:

```sh
cargo build --release
```

The executables will be generated in `target/release` directory. For usage
information call each executable with the `--help` argument.

To run some of the experiments described in the paper [Data Structures for
Direct Spanning Tree Representations in Mutation-based Evolutionary
Algorithms][ieee-paper] execute `make` in the `experiments` directory (it
requires [Python 3][python], [GNU Parallel][parallel] and [Tectonic][tectonic].

## License

Licensed under [Mozilla Public License 2.0][mpl]. Contributions will be
accepted under the same license.

[ieee-paper]: https://ieeexplore.ieee.org/document/8764437
[install-rust]: https://www.rust-lang.org/install.html
[mpl]: https://www.mozilla.org/en-US/MPL/2.0/
[parallel]: https://www.gnu.org/software/parallel/
[python]: https://python.org
[tectonic]: https://tectonic-typesetting.github.io/
