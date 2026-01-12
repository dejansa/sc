# sc

Simple command-line utility for plotting sensor data from SC export files.

## _**Installation**_

### I recommend [uv](https://docs.astral.sh/uv/):
```bash
$ uv tool install git+https://github.com/dejansa/sc.git

# To upgrade to latest version run:
$ uv tool upgrade sc
# or
$ uv tool upgrade --all
```

### [pipx](https://github.com/pypa/pipx) can also be used:
```bash
$ pipx install git+https://github.com/dejansa/sc.git

# To upgrade to latest version run:
$ pipx upgrade sc

# or

$ pipx upgrade-all
```

## _**Usage**_

```bash
❯ scx --help
usage: scx [-h] [-g GROUP] [-v] [-ma MA_WINDOW] file

Plot SC sensor export (TSV/CSV) traces by column group

positional arguments:
  file                  Path to the CSV or TSV file to parse

options:
  -h, --help            show this help message and exit
  -g GROUP, --group GROUP
                        Comma-separated column groups to visualize
                        (valid: ang, acc, as, h, angd, accn, asn, hn, tilt, mad, mah, ekf; default: acc)
                        ang  - angles
                        acc  - accelerations
                        as   - angular speed (gyroscope)
                        h    - magnetic field (magnetometer)
                        angd - sensor1 - sensor2 angle delta
                        accn - acceleration magnitude (norm)
                        asn  - angular speed magnitude (norm)
                        hn   - magnetic field magnitude (norm)
                        tilt - pitch/roll from accelerometer + yaw heading
                        mad  - Madgwick filter (MARG if magnetometer present)
                        mah  - Mahony filter (MARG if magnetometer present)
                        ekf  - Extended Kalman Filter (MARG if magnetometer present)
  -v, --version         Show the installed package version
  -ma MA_WINDOW, --ma-window MA_WINDOW
                        Simple moving average window size per signal (default: 5)
```

### **_example_**

![log](assets/example1.png)
