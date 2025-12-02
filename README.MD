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
| `-t` | `--sigma-t` | Disorder variance for hopping (default: 0.0) |
| `-V` | `--sigma-U` | Disorder variance for interaction (default: 0.0) |
| `-v` | `--sigma-u` | Disorder variance for potential (default: 0.0) |
| `-h` | `--help` | Display usage information |

## Example

```bash
./BoseGlass -m 8 -n 8 -T 0.01 -U 0.01 -u 0 -r 100 -s 5 -f "u"
```

