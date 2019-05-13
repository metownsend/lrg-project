# makeCutoutTableFunc.py
# function to make a table of cutouts scrapped from DECaLS and SDSS

def make_cutout_comparison_table(ra, dec, objid, z, specobjid, brickid): # tag, pixel):
    
    from astropy.io import ascii
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    de_cutout_url = 'http://legacysurvey.org/viewer/jpeg-cutout/?ra={0.ra.deg}&dec={0.dec.deg}&layer=decals-dr7&pixscale=0.1&bands=grz'
    mod_cutout_url = 'http://legacysurvey.org/viewer/jpeg-cutout/?ra={0.ra.deg}&dec={0.dec.deg}&layer=decals-dr7-model&pixscale=0.1&bands=grz'
    resid_cutout_url = 'http://legacysurvey.org/viewer/jpeg-cutout/?ra={0.ra.deg}&dec={0.dec.deg}&layer=decals-dr7-resid&pixscale=0.1&bands=grz'
    sd_cutout_url = 'http://legacysurvey.org/viewer/jpeg-cutout/?ra={0.ra.deg}&dec={0.dec.deg}&layer=sdssco&pixscale=0.1&bands=gri'
    dviewurl = []
    dmodviewurl = []
    dresidviewurl = []
    sviewurl = []
    tabrows = []
    deimg = []
    sdimg = []
    redshift = []
		
    for i in range(len(ra)):
        dviewurl.append('http://legacysurvey.org/viewer?ra={}&dec={}&zoom=15&layer=decals-dr7'.format(ra[i], dec[i]))
        dmodviewurl.append('http://legacysurvey.org/viewer?ra={}&dec={}&zoom=15&layer=decals-dr7-model'.format(ra[i], dec[i]))
        dresidviewurl.append('http://legacysurvey.org/viewer?ra={}&dec={}&zoom=15&layer=decals-dr7-resid'.format(ra[i], dec[i]))
        sviewurl.append('http://skyserver.sdss.org/dr14/en/tools/chart/navi.aspx?ra={}&dec={}'.format(ra[i], dec[i]))
		
    
    for i in range(len(ra)):
        specurl = 'http://skyserver.sdss.org/dr14/en/get/SpecById.ashx?id={}'.format(specobjid[i])
        sc = SkyCoord(ra[i], dec[i], unit=u.deg)
        deimg = '<a href="{}"><img src="{}"></a>'.format(dviewurl[i], de_cutout_url.format(sc))
        modimg = '<a href="{}"><img src="{}"></a>'.format(dmodviewurl[i], mod_cutout_url.format(sc))
        residimg = '<a href="{}"><img src="{}"></a>'.format(dresidviewurl[i], resid_cutout_url.format(sc))
        sdimg = '<a href="{}"><img src="{}"></a>'.format(sviewurl[i], sd_cutout_url.format(sc))
        # file = '<a href="{}"><img src="/Users/mtownsend/Documents/Cutouts/Radius_Cutouts/Cutouts/{}.jpg"></a>'.format(dviewurl[i], objid[i])
        file = '<a href="{}"><img src="/Users/mtownsend/anaconda/GitHub/lrg-project/Cutouts/{}-{}.jpg"></a>'.format(dviewurl[i], brickid[i], objid[i])
        info = '{}-{}<br>z={:.3f}<br><br><a href="{}">spectrum</a>'.format(brickid[i], objid[i], z[i], specurl)
# 		info = '{}-{}<br>z={:.3f}<br><br><a href="{}">spectrum</a>'.format(brickid_match[i], objstr[i], z[i], specurl)
# 		info = '{}-{}<br>z={:.3f}<br><br>tag={}<br>pixel={} <br><a href="{}">spectrum</a>'.format(brickid[i], objid[i], z[i], tag[i], pixel[i], specurl)
        tabrows.append('<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(info, deimg, modimg, residimg, sdimg, file))

		
    htmlstr = """
	<table border="2" width="30%"><tbody>
	<tr>

	<tr>
	<th>Object ID and Redshift</th>
	<th>DECaLS</th>
	<th>Model</th>
	<th>Residual</th>
	<th>SDSS</th>
	# <th>Radii</th>
	</tr>

	{}
	</table>
	""".format('\n'.join(tabrows))
    
    return htmlstr