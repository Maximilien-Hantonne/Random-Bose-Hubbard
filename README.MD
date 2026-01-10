# Random Bose-Hubbard Model

Exact diagonalization of the Bose-Hubbard Hamiltonian with disorder


## Options

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-m` | `--sites` | Number of lattice sites |
| `-n` | `--bosons` | Number of bosons |
| `-t` | `--hopping` | Hopping parameter t (initial value) |
| `-U` | `--interaction` | On-site interaction parameter U (initial value) |
| `-u` | `--potential` | Chemical potential μ (initial value) |
| `-a` | `--range-t` | Range for hopping parameter t |
| `-b` | `--range-U` | Range for interaction parameter U |
| `-c` | `--range-u` | Range for chemical potential u |
| `-A` | `--step-t` | Step size for hopping parameter t |
| `-B` | `--step-U` | Step size for interaction parameter U |
| `-C` | `--step-u` | Step size for chemical potential u |
| `-f` | `--fixed` | Fixed parameter (`t`, `U`, or `u`) |
| `-S` | `--scale` | Spacing scale: `lin` (linear) or `log` (logarithmic) (default : log) |
| `-d` | `--distrib` | Disorder distribution: `uni` (uniform) or `gaus` (gaussian) (default : uni)|
| `-T` | `--delta-t` | Relative disorder half-width or standard deviation for hopping (default: 0.0) |
| `-V` | `--delta-U` | Relative disorder half-width or standard deviation for interaction (default: 0.0) |
| `-v` | `--delta-u` | Relative disorder half-width or standard deviation for potential (default: 0.0) |
| `-R` | `--realizations` | Number of disorder realizations for averaging (default: 1) |
| `-h` | `--help` | Display usage information |


## Example

```bash
# Without disorder - logarithmic scale (default)
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f "u"

# Linear scale
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f "u" -S "lin"

# With uniform disorder averaging over 10 realizations
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f "u" -T 0.5 -R 10

# With Gaussian disorder
./RBH -m 8 -n 8 -t 0.01 -U 0.1 -u 1 -a 1 -b 8 -A 0.01 -B 0.1 -f "u" -T 3.75 -R 10 -d "gaus"
```

## Authors

- [Maximilien HANTONNE](https://github.com/Maximilien-Hantonne)


## License

This project is licensed under the GNU GPLv3 - see [LICENCE](https://github.com/Maximilien-Hantonne/Random-Bose-Hubbard/blob/master/LICENSE) file for details.
