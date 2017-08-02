#!/bin/bash

# Valid Train classes
# Bomber               : 229
# Leggings             : 3571
# Skirt                : 10794
# Joggers              : 3260
# Kimono               : 1637
# Cutoffs              : 1177
# Parka                : 491
# Shorts               : 14195
# Cardigan             : 9606
# Coat                 : 1539
# Trunks               : 287
# Poncho               : 579
# Tank                 : 11204
# Jacket               : 7548
# Robe                 : 107
# Jersey               : 534
# Jeggings             : 443
# Henley               : 521
# Turtleneck           : 99
# Flannel              : 224
# Chinos               : 374
# Sweatshorts          : 781
# Jumpsuit             : 4464
# Top                  : 7270
# Jeans                : 5126
# Button-Down          : 243
# Blouse               : 17752
# Blazer               : 5408
# Hoodie               : 2910
# Culottes             : 359
# Sweatpants           : 2224
# Anorak               : 121
# Kaftan               : 98
# Romper               : 5425
# Sweater              : 9517


# Invalid Train classes due to less or very large data count
# Capris               : 57
# Sarong               : 18
# Nightdress           : 0
# Shirtdress           : 0
# Cape                 : 0
# Coverup              : 13
# Onesie               : 47
# Jodhpurs             : 32
# Halter               : 11
# Sundress             : 0
# Caftan               : 38
# Gauchos              : 35
# Peacoat              : 63
# Tee                  : 26653
# Dress                : 52138

rm -rf dataset/train/Capris
rm -rf dataset/validation/Capris
rm -rf dataset/test/Capris
rm -rf dataset/train/Sarong
rm -rf dataset/validation/Sarong
rm -rf dataset/test/Sarong
rm -rf dataset/train/Nightdress
rm -rf dataset/validation/Nightdress
rm -rf dataset/test/Nightdress
rm -rf dataset/train/Shirtdress
rm -rf dataset/validation/Shirtdress
rm -rf dataset/test/Shirtdress
rm -rf dataset/train/Cape
rm -rf dataset/validation/Cape
rm -rf dataset/test/Cape
rm -rf dataset/train/Coverup
rm -rf dataset/validation/Coverup
rm -rf dataset/test/Coverup
rm -rf dataset/train/Onesie
rm -rf dataset/validation/Onesie
rm -rf dataset/test/Onesie
rm -rf dataset/train/Jodhpurs
rm -rf dataset/validation/Jodhpurs
rm -rf dataset/test/Jodhpurs
rm -rf dataset/train/Halter
rm -rf dataset/validation/Halter
rm -rf dataset/test/Halter
rm -rf dataset/train/Sundress
rm -rf dataset/validation/Sundress
rm -rf dataset/test/Sundress
rm -rf dataset/train/Caftan
rm -rf dataset/validation/Caftan
rm -rf dataset/test/Caftan
rm -rf dataset/train/Gauchos
rm -rf dataset/validation/Gauchos
rm -rf dataset/test/Gauchos
rm -rf dataset/train/Peacoat
rm -rf dataset/validation/Peacoat
rm -rf dataset/test/Peacoat
rm -rf dataset/train/Tee
rm -rf dataset/validation/Tee
rm -rf dataset/test/Tee
rm -rf dataset/train/Dress
rm -rf dataset/validation/Dress
rm -rf dataset/test/Dress


