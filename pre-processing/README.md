# Pre-processing

This process consists of removing noise from the hydrated tweets. In particular these scripts provided in this folder remove noise such as:
- Leading usernames
- Emojis and emoticons
- Unwanted characters (such as URLs and hashtags)

For the pre-processing part, here's an example showing how to execute the script provided (for the datasets without noise).

```console
!python pre-processing.py 1.0 17 1
```
Where:

- The first parameter represents if we want to split our training data (by default we didn't split it, so we leave it to 100% or 1.0)
- The second and third parameter represents the random seed and the random state (this will only work if we want to split our training data)

**NOTE: ** It is important to have the generated files from the hydration process in the same folder level with respect to this script.
