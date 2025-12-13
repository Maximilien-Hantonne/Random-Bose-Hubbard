# Random Bose-Hubbard Model

Exact diagonalization of the Bose-Hubbard Hamiltonian with disorder


## Options

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-m` | `--sites` | Number of lattice sites |
| `-n` | `--bosons` | Number of bosons |
| `-t` | `--hopping` | Hopping parameter t |
| `-U` | `--interaction` | On-site interaction parameter U |
| `-u` | `--potential` | Chemical potential μ |
| `-r` | `--range` | Range for parameter sweep |
| `-s` | `--step` | Step size for parameter sweep |
| `-f` | `--fixed` | Fixed parameter (`t`, `U`, or `u`) |
| `-T` | `--sigma-t` | Relative disorder variance for hopping (Gaussian, default: 0.0) |
| `-V` | `--delta-U` | Relative disorder half-width for interaction (Uniform, default: 0.0) |
| `-v` | `--delta-u` | Relative disorder half-width for potential (Uniform, default: 0.0) |
| `-R` | `--realizations` | Number of disorder realizations for averaging (default: 1) |
| `-h` | `--help` | Display usage information |

## Example

```bash
# Without disorder
./BoseGlass -m 8 -n 8 -T 0.01 -U 0.01 -u 0 -r 100 -s 5 -f "u"

# With disorder averaging over 10 realizations
./BoseGlass -m 8 -n 8 -t 0.01 -U 0.01 -u 0 -r 100 -s 5 -f "u" -T 0.5 -R 10
```

