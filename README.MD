# Random Bose-Hubbard Model

<div align="center">
    <a href="https://github.com/Maximilien-Hantonne/Random-Bose-Hubbard">
        <img src="https://github.com/Maximilien-Hantonne/Random-Bose-Hubbard/blob/master/figures/off_diagonal_on_diagonal_disorder.JPG" alt="Random-Bose-Hubbard phase diagram" width="350">
    </a>
    <h3 align="center"> Transitions of the Random Bose Hubbard model</h3>
    <p align="center">
        An implementation of the Random Bose Hubbard model with exact diagonalization to characterize the phase diagram of the system.
    </p>
</div>


## Options

### System Size

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-m` | `--sites` | Number of lattice sites |
| `-n` | `--bosons` | Number of bosons |

### Model Parameters

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-t` | `--hopping` | Hopping parameter t |
| `-U` | `--interaction` | On-site interaction parameter U |
| `-u` | `--chemical-potential` | Chemical potential μ |

### Parameter Scans

| Long Form | Description |
|-----------|-------------|
| `--tr` | Range for hopping parameter t |
| `--Ur` | Range for interaction parameter U |
| `--ur` | Range for chemical potential μ |
| `--ts` | Step size for hopping parameter t |
| `--Us` | Step size for interaction parameter U |
| `--us` | Step size for chemical potential μ |

### Disorder

| Long Form | Description |
|-----------|-------------|
| `--tD` | Disorder strength for t (default: 0.0) |
| `--UD` | Disorder strength for U (default: 0.0) |
| `--uD` | Disorder strength for μ (default: 0.0) |
| `--tp` | Probability distribution for t: `uni` or `gaus` (default: uni) |
| `--Up` | Probability distribution for U: `uni` or `gaus` (default: uni) |
| `--up` | Probability distribution for μ: `uni` or `gaus` (default: uni) |
| `-R`, `--realizations` | Number of disorder realizations for averaging (default: 1) |

### Other Options

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-f` | `--fixed` | Fixed parameter (`t`, `U`, or `u`) |
| `-S` | `--scale` | Spacing scale: `lin` or `log` (default: log) |
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
- `--tD 0.0:0.1:0.5` → disorder: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
- `-R 1:5:21` → realizations: 1, 6, 11, 16, 21

**Note:** No quotes needed for comma-separated nor range syntax.


## Example

```bash
# Without disorder - logarithmic scale (default)
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 --tr 10 --Ur 10 --ts 0.5 --Us 0.5 -f u

# Linear scale
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 --tr 10 --Ur 10 --ts 0.5 --Us 0.5 -f u -S lin

# With uniform disorder averaging over 10 realizations
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 --tr 10 --Ur 10 --ts 0.5 --Us 0.5 -f u --tD 0.5 -R 10

# With Gaussian disorder on t only
./RBH -m 8 -n 8 -t 0.01 -U 0.1 -u 1 --tr 1 --Ur 8 --ts 0.01 --Us 0.1 -f u --tD 3.75 -R 10 --tp gaus

# With different distributions: Gaussian for t, uniform for U
./RBH -m 8 -n 8 -t 0.01 -U 0.1 -u 1 --tr 1 --Ur 8 --ts 0.01 --Us 0.1 -f u --tD 0.5 --UD 0.3 -R 10 --tp gaus --Up uni

# Using comma-separated list - runs for 3 disorder values
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 --tr 10 --Ur 10 --ts 0.5 --Us 0.5 -f u --uD 0.3,0.5,0.7 -R 10

# Using range syntax - sweep disorder from 0.0 to 1.0 in steps of 0.2
./RBH -m 8 -n 8 -t 0.01 -U 0.01 -u 0 --tr 10 --Ur 10 --ts 0.5 --Us 0.5 -f u --uD 0.0:0.2:1.0 -R 10

# Multiple parameters with lists - iterates by index (2 runs total)
./RBH -m 8,9 -n 8,9 -t 0.01 -U 0.01 -u 0 --tr 10 --Ur 10 --ts 0.5 --Us 0.5 -f u --uD 0.1,0.3 -R 5

# System size sweep
./RBH -m 6:1:10 -n 6:1:10 -t 0.01 -U 0.01 -u 0 --tr 10 --Ur 10 --ts 0.5 --Us 0.5 -f u
```

## Authors

[Maximilien HANTONNE](https://github.com/Maximilien-Hantonne)


## License

This project is licensed under the GNU GPLv3 - see [LICENCE](https://github.com/Maximilien-Hantonne/Random-Bose-Hubbard/blob/master/LICENSE) file for details.
