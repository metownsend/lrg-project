from pythonds.basic.stack import Stack
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u


# Read in data
hdulist = fits.open('/home/mtownsend/anaconda3/Data/survey-dr3-specObj-dr13.fits') # survey
hdulist2 = fits.open('/home/mtownsend/anaconda3/Data/specObj-dr13.fits') # sdss
tbdata = hdulist[1].data
tbdata2 = hdulist2[1].data


# Object ID from survey file; value -1 for non-matches
objid = []
objid = tbdata.field('OBJID')
# print(len(objid)) 

# RA
ra = []
ra = tbdata.field('RA')
# print(type(ra))
# print(len(ra))

# Dec
dec = []
dec = tbdata.field('DEC')
# print(len(dec))

# Class of object
gal_class = []
gal_class = tbdata2.field('CLASS')
# print(len(gal_class))

# What survey the data is from
survey = []
survey = tbdata2.field('SURVEY')
# print(len(survey))

# surveytemp = []
# 
# for x in range(len(survey)):
# 	surveytemp.append(survey[x])
# 	
# survey_array = np.array(surveytemp)


# SPECPRIMARY; set to 1 for primary observation of object, 0 otherwise
spec = []
spec = tbdata2.field('SPECPRIMARY')
# print(len(spec))

# Bitmask of spectroscopic warning values; need set to 0
zwarn_noqso = []
zwarn_noqso = tbdata2.field('ZWARNING_NOQSO')
# print(len(zwarn_noqso))

# Spectroscopic classification for certain redshift?
class_noqso = []
class_noqso = tbdata2.field('CLASS_NOQSO')
# print(len(class_noqso))

# Type of galaxy according to DECaLS
type = []
type = tbdata.field('TYPE')
# print(len(type))

# Array for LOWZ targets
targets = []
# target_match = []
targets = tbdata2.field('BOSS_TARGET1')
# target_match = targets[np.where((objid > -1) & (gal_class == 'GALAXY') & (spec == 1 ) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))]
# print(len(targets))

# targetVec = [0, 24, 67, 75]

# ----------------------------------------------------------------------------------
# Function to find LOWZ targets

def divideBy2(decNumber):

	np.vectorize(decNumber)
	remstack = Stack()
	
	if decNumber == 0: return "0"
	
	while decNumber > 0:
		rem = decNumber % 2
		remstack.push(rem)
		decNumber = decNumber // 2
		
	binString = ""
	while not remstack.isEmpty():
		binString = binString + str(remstack.pop())
			
	return binString
	
# ----------------------------------------------------------------------------------

divideBy2Vec = np.vectorize(divideBy2)

a = divideBy2Vec(targets) # gives binary in string form
# print(a[0,0])
bin2int = [int(i) for i in a] # converts binary strings to integer
# print(bin2int)
tar = np.array(bin2int) # puts list of integers into numpy array
c = tar % 2 # divide by two again to see if the binary number ends in zero
# print(type(c))
lowz_tar = np.array(c)
# lowz_tar = tar[np.where(c == 1)] # the targets we want have a remainder 1
# print(len(lowz_tar))

# print(lowz_tar)


# --------------------------------------------------------------------------------------
# Find RA an Dec of LOWZ targets

ramatch = ra[np.where((lowz_tar == 1) & (ra >= 241) & (ra <= 246) & (dec >= 6.5) & (dec <= 11.5) & (objid > -1) & (gal_class == 'GALAXY') & (spec == 1 ) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')) & ((type == 'DEV') | (type == 'EXP') | (type == 'COMP')))]

decmatch = dec[np.where((lowz_tar == 1) & (ra >= 241) & (ra <= 246) & (dec >= 6.5) & (dec <= 11.5) & (objid > -1) & (gal_class == 'GALAXY') & (spec == 1 ) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')) & ((type == 'DEV') | (type == 'EXP') | (type == 'COMP')))]


de_cutout_url = 'http://legacysurvey.org/viewer/jpeg-cutout/?ra={}&dec={}&layer=decals-dr3&pixscale=0.1&bands=grz'
sd_cutout_url = 'http://legacysurvey.org/viewer/jpeg-cutout/?ra={}&dec={}&layer=sdssco&pixscale=0.1&bands=gri'
dviewurl = []
sviewurl = []
tabrows = []
deimg = []
sdimg = []
    
for i in range(len(ramatch)):
	dviewurl.append('http://legacysurvey.org/viewer?ra={}&dec={}'.format(ramatch[i], decmatch[i]))
	sviewurl.append('http://skyserver.sdss.org/dr12/en/tools/chart/navi.aspx?ra={}&dec={}'.format(ramatch[i], decmatch[i]))
# 	sc = SkyCoord(ramatch[i], decmatch[i], unit=u.deg)
	deimg = '<a href="{}"><img src="{}"></a>'.format(dviewurl, de_cutout_url.format(ramatch[i], decmatch[i]))
	sdimg = '<a href="{}"><img src="{}"></a>'.format(sviewurl, sd_cutout_url.format(ramatch[i], decmatch[i]))


htmlstr = """
<html><body>
<h1>DECaLS image, Model, Residual, SDSS image</h1>
<table border="2" width="30%"><tbody>
<tr>
<td><img src=de_cutout_url></a></td>
<td><img src=sd_cutout_url></a></td>
</tr>
</tbody></table>
</body></html>
"""

Html_file= open("cutout_website.html","w")
Html_file.write(htmlstr)
Html_file.close()