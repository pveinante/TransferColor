# Color Transfer

Meven LEBLANC and Paul VEINANTE work about color transfer for digital imaging in ESIR. 
See resume [here](ColorTransferResume.pdf)

## LDR Color Transfer

Python implementation of Color Transfer between images. 
See code [here](src/Reinhard.py)

### Usage: Reinhard.py [OPTIONS] COMMAND [ARGS]...

### Commands:
  - colortransfer 
  - help

## HDR Color Transfer

Color transfer for Extension for HDR images. 
See code [here](src/HDR_Reinhard.py)

### Usage: HDR_Reinhard.py [OPTIONS] COMMAND [ARGS]...

### Commands:
  - colortransfer
  - help

## MGD 

Color transfer using MDG.
See code [here](src/base_colortransfer.py)


## MGD 

Color transfer using MDG.
See code [here](src/base_colortransfer.py)

### Usage: base_colortransfer.py [OPTIONS] COMMAND [ARGS]...

Write line 10 and 11 the source and target images.

### Usage: base_colortransfer.py simplegaussian

### Commands:
  - colortransfer                      (Execute Reinhard color transfer)
  - multivariategaussian               (Execute MGD color transfer)
  - multivariategaussian --chomaonly   (Execute MGD with particular luminance processing color transfer)
  
