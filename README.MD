# Random Bose-Hubbard Model

Exact diagonalization of the Bose-Hubbard Hamiltonian with disorder


## Options

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-m` | `--sites` | Number of lattice sites |
| `-n` | `--bosons` | Number of bosons |
| `-T` | `--hopping` | Hopping parameter T |
| `-U` | `--interaction` | On-site interaction parameter U |
| `-u` | `--potential` | Chemical potential μ |
| `-r` | `--range` | Range for parameter sweep |
| `-s` | `--step` | Step size for parameter sweep |
| `-f` | `--fixed` | Fixed parameter (`T`, `U`, or `u`) |
| `-t` | `--sigma-t` | Relative disorder variance for hopping (default: 0.0) |
| `-V` | `--sigma-U` | Relative disorder variance for interaction (default: 0.0) |
| `-v` | `--sigma-u` | Relative disorder variance for potential (default: 0.0) |
| `-R` | `--realizations` | Number of disorder realizations for averaging (default: 1) |
| `-h` | `--help` | Display usage information |

## Example

```bash
# Without disorder
./BoseGlass -m 8 -n 8 -T 0.01 -U 0.01 -u 0 -r 100 -s 5 -f "u"

# With disorder averaging over 10 realizations
./BoseGlass -m 8 -n 8 -T 0.01 -U 0.01 -u 0 -r 100 -s 5 -f "u" -t 0.5 -R 10
```

