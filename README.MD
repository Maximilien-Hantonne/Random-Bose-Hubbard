# Random Bose-Hubbard Model

Exact diagonalization of the Bose-Hubbard Hamiltonian with or without disorder


## Options

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-m` | `--sites` | Number of lattice sites |
| `-n` | `--bosons` | Number of bosons |
| `-t` | `--hopping` | Hopping parameter t initial value |
| `-U` | `--interaction` | On-site interaction parameter U initial value |
| `-u` | `--potential` | Chemical potential μ initial value |
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


## Multi-Value Parameters

All numeric parameters support three formats for specifying multiple values. When multiple values are given, the code runs consecutively for each combination.

### Formats

| Format | Syntax | Example | Result |
|--------|--------|---------|--------|
| **Single value** | `value` | `-v 0.3` | `0.3` |
| **Comma-separated** | `v1,v2,v3` | `-v 0.3,0.5,0.7` | `0.3, 0.5, 0.7` |
| **Range** | `start:step:end` | `-v 0.1:0.2:0.9` | `0.1, 0.3, 0.5, 0.7, 0.9` |

### Range Syntax

The range syntax `start:step:end` generates values from `start` to `end` (inclusive) with increment `step`:
- `-m 6:2:10` → sites: 6, 8, 10
- `-v 0.0:0.1:0.5` → disorder: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
- `-R 1:5:21` → realizations: 1, 6, 11, 16, 21

**Note:** No quotes needed for comma-separated nor range syntax.


## Example

```bash
# Without disorder - logarithmic scale (default)
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f u

# Linear scale
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f u -S lin

# With uniform disorder averaging over 10 realizations
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f u -T 0.5 -R 10

# With Gaussian disorder
./RBH -m 8 -n 8 -t 0.01 -U 0.1 -u 1 -a 1 -b 8 -A 0.01 -B 0.1 -f u -T 3.75 -R 10 -d gaus

# Using comma-separated list - runs for 3 disorder values
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f u -v 0.3,0.5,0.7 -R 10

# Using range syntax - sweep disorder from 0.0 to 1.0 in steps of 0.2
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f u -v 0.0:0.2:1.0 -R 10

# Multiple parameters with ranges - runs for all combinations (2 × 3 = 6 runs)
./RBH -m 8,9 -n 8,9 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f u -v 0.1:0.2:0.5 -R 5

# System size sweep
./RBH -m 6:1:10 -n 6:1:10 -t 0.01 -U 0.01 -u 0 -a 10 -b 10 -A 0.5 -B 0.5 -f u
```

## Authors

- [Maximilien HANTONNE](https://github.com/Maximilien-Hantonne)


## License

This project is licensed under the GNU GPLv3 - see [LICENCE](https://github.com/Maximilien-Hantonne/Random-Bose-Hubbard/blob/master/LICENSE) file for details.
