# gax2fur
a GAX ROM to Impulse Tracker Module (.it) Converter

NOTE: there may be some inaccuracies with this converter

### usage
**make sure you have an empty folder called "out" in the repo folder**

an example command to get converted modules out of a GBA rom
```
python3 gax2it.py --file_path example_rom.gba
```
the converted .it files will be in the now not empty "out" folder

### credits
[shinen-gax-python by beanieaxolotl](https://github.com/beanieaxolotl/shinen-gax-python) for the original GAX Python library used in this project. I modified it a little bit to also support GAX v1 and v2 tunes alongside GAX v3 tunes.
