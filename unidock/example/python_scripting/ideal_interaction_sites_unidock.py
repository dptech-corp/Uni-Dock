#!/usr/bin/python

# requires Maestro or Amber atom names (e.g. HNE or HE, HH11, HH12, HH21 and HH22 for Arg hydrogens in the guanidinium group)
# requires biopython for PDB parsing
# for h-bond calculation give histidine protonation explicit: HIE, HID or HIP

# import packages
import sys
import math as mt
import numpy as np
from Bio.PDB import *

# distance (A) between O/N---H in hydrogen bonds
HBOND_DIST = 1.9
# distance (A) between planes in aromatic interactions
AROM_DIST = 3.8

def optparser(argv=sys.argv[1:]):
	
	''' Returns a dictionary "args" with pdb file name, chain ID,
		and residue IDs to analyze
	'''
	
	# usage: input line
	usage = '''
	Usage:

	python ideal_interaction_sites.py -i pdb_file -hb chainID:resID:atomType[,chainID:resID:atomType]

		-i  : input PDB file name containing the protein structure (with 
		     hydrogens)
		-hb : each "chainID:resID:atomType" pair identifies an atom, which is either H-bond 
			 donor or acceptor, different atoms are separated by commar
		  chainID   : protein chain ID (e.g. A)
		  resID     : residues ID (e.g. 24)
		  atomType  : the unique atom type in PDB file (e.g. O)

		(optional) -d : output directory
	
	Notes:
	
	- python requires biopython
	- Requires Maestro or Amber atom names for Hs (e.g. HNE or HE, HH11,
	  HH12, HH21 and HH22 for Arg hydrogens in the guanidinium group).
	- Requires Biopython for PDB parsing.
	- For H bond calculations assign histidine protonation explicitly
	  and name residue as HIE, HID or HIP.
	'''
	
	# initiate dictionary with arguments
	args = dict()
	
	# test number of arguments
	if len(argv) < 4:
		print('Error: invalid number of arguments')
		print('Requires -i and -hb arguments (check "chainID:resID:atomType"s are comma separated)')
		print(usage)
		sys.exit(2)
	
	# add pdb input file name to dictionary
	if '-i' in argv:
		pdb_file_index = argv.index('-i') + 1
		args.update({'pdb_file' : argv[pdb_file_index]})
	else:
		print('Error: -i argument required indicating pdb file')
		print(usage)
		sys.exit(2)		
	
	# add hb to dictionary
	if '-hb' in argv:
		hb_index = argv.index('-hb') + 1
		args.update({'hbs' : argv[hb_index].split(",")})
	else:
		print('Error: -hb argument required indicating H-bond atom(s)')
		print(usage)
		sys.exit(2)		

	if '-d' in argv:
		dir_index = argv.index('-d') + 1
		args.update({'dir' : argv[dir_index]})
	
	# output dictionary with bias_file_name, gpf_file_name, dpf_file_name and/or dpf_directory_name
	return args


# define ideal H bond interactions for protein residues (side chains)
def ideal_interactions(HBOND_DIST):
	
	''' Returns a list of lists with ideal H bond interaction definitions:
		distance, angle, diherdal.
	'''
	
	# side chain interactions (acceptors)
	# sintax: [resname,[atom1,atom2,atom3],[distance,angle,dihedral],type]
	sc_acceptors = [['GLU',['CG','CD','OE1'],[HBOND_DIST,180,0],'acc'],
                    ['GLU',['CG','CD','OE1'],[HBOND_DIST,210,0],'acc'],
                    ['GLU',['CG','CD','OE1'],[HBOND_DIST,240,0],'acc'],
                    ['GLU',['CG','CD','OE1'],[HBOND_DIST,150,0],'acc'],
                    ['GLU',['CG','CD','OE1'],[HBOND_DIST,120,0],'acc'],

                    ['GLU',['CG','CD','OE2'],[HBOND_DIST,180,0],'acc'],
                    ['GLU',['CG','CD','OE2'],[HBOND_DIST,210,0],'acc'],
                    ['GLU',['CG','CD','OE2'],[HBOND_DIST,240,0],'acc'],
                    ['GLU',['CG','CD','OE2'],[HBOND_DIST,150,0],'acc'],
                    ['GLU',['CG','CD','OE2'],[HBOND_DIST,120,0],'acc'],

                    ['ASP',['CB','CG','OD1'],[HBOND_DIST,180,0],'acc'],
                    ['ASP',['CB','CG','OD1'],[HBOND_DIST,210,0],'acc'],
                    ['ASP',['CB','CG','OD1'],[HBOND_DIST,240,0],'acc'],
                    ['ASP',['CB','CG','OD1'],[HBOND_DIST,150,0],'acc'],
                    ['ASP',['CB','CG','OD1'],[HBOND_DIST,120,0],'acc'],

                    ['ASP',['CB','CG','OD2'],[HBOND_DIST,180,0],'acc'],
                    ['ASP',['CB','CG','OD2'],[HBOND_DIST,210,0],'acc'],
                    ['ASP',['CB','CG','OD2'],[HBOND_DIST,240,0],'acc'],
                    ['ASP',['CB','CG','OD2'],[HBOND_DIST,150,0],'acc'],
                    ['ASP',['CB','CG','OD2'],[HBOND_DIST,120,0],'acc'],

                    ['ASN',['CB','CG','OD1'],[HBOND_DIST,180,0],'acc'],
                    ['ASN',['CB','CG','OD1'],[HBOND_DIST,210,0],'acc'],
                    ['ASN',['CB','CG','OD1'],[HBOND_DIST,240,0],'acc'],
                    ['ASN',['CB','CG','OD1'],[HBOND_DIST,150,0],'acc'],
                    ['ASN',['CB','CG','OD1'],[HBOND_DIST,120,0],'acc'],

                    ['GLN',['CG','CD','OE1'],[HBOND_DIST,180,0],'acc'],
                    ['GLN',['CG','CD','OE1'],[HBOND_DIST,210,0],'acc'],
                    ['GLN',['CG','CD','OE1'],[HBOND_DIST,240,0],'acc'],
                    ['GLN',['CG','CD','OE1'],[HBOND_DIST,150,0],'acc'],
                    ['GLN',['CG','CD','OE1'],[HBOND_DIST,120,0],'acc'],

                    ['SER',['CA','CB','OG'], [HBOND_DIST,109.5,'alpha'],'acc'],
                    ['SER',['CA','CB','OG'], [HBOND_DIST,109.5,'beta'], 'acc'],

                    ['THR',['CA','CB','OG1'],[HBOND_DIST,109.5,'alpha'],'acc'],
                    ['THR',['CA','CB','OG1'],[HBOND_DIST,109.5,'beta'], 'acc'],

                    ['TYR',['CE1','CZ','OH'],[HBOND_DIST,109.5,'alpha'],'acc'],
                    ['TYR',['CE1','CZ','OH'],[HBOND_DIST,109.5,'beta'], 'acc'],

                    ['HID',['CG','CD2','NE2'], [HBOND_DIST,125.4,180],'acc'],
                    ['HIE',['NE2','CE1','ND1'],[HBOND_DIST,125.4,180],'acc'],
                    
                    ['HOH',['H1','H2','O'],    [HBOND_DIST,109.5,120],'acc'],
                    ['HOH',['H1','H2','O'],    [HBOND_DIST,109.5,240],'acc'],
                    ['WAT',['H1','H2','O'],    [HBOND_DIST,109.5,120],'acc'],
                    ['WAT',['H1','H2','O'],    [HBOND_DIST,109.5,240],'acc']]

	# side chain interactions (donors)
	# sintax: [resname,[atom1,atom2,atom3],[distance,angle,dihedral],type]
	sc_donors = [['ASN',['CG','ND2','HD21'],[HBOND_DIST,180,0],'don'],
                 ['ASN',['CG','ND2','HD22'],[HBOND_DIST,180,0],'don'],

                 ['GLN',['CD','NE2','HE21'],[HBOND_DIST,180,0],'don'],
		         ['GLN',['CD','NE2','HE22'],[HBOND_DIST,180,0],'don'],

                 ['ARG',['CZ','NE','HNE'],  [HBOND_DIST,180,0],'don'],
                 ['ARG',['CZ','NH1','HH11'],[HBOND_DIST,180,0],'don'],
                 ['ARG',['CZ','NH1','HH12'],[HBOND_DIST,180,0],'don'],
                 ['ARG',['CZ','NH2','HH21'],[HBOND_DIST,180,0],'don'],
                 ['ARG',['CZ','NH2','HH22'],[HBOND_DIST,180,0],'don'],
                 ['ARG',['CZ','NE','HE'],   [HBOND_DIST,180,0],'don'],

                 ['SER',['CB','OG','HG'],[HBOND_DIST,180,0],'don'],

                 ['THR',['CB','OG1','HG1'],[HBOND_DIST,180,0],'don'],

                 ['TYR',['CZ','OH','HH'],[HBOND_DIST,180,0],'don'],

                 ['TRP',['CD1','NE1','HE1'],[HBOND_DIST,180,0],'don'],

                 ['LYS',['CE','NZ','HZ1'],[HBOND_DIST,180,0],'don'],
                 ['LYS',['CE','NZ','HZ2'],[HBOND_DIST,180,0],'don'],
                 ['LYS',['CE','NZ','HZ3'],[HBOND_DIST,180,0],'don'],

                 ['HIE',['CE1','NE2','HE2'],[HBOND_DIST,180,0],'don'],
                 ['HID',['CE1','ND1','HD1'],[HBOND_DIST,180,0],'don'],
                 ['HIP',['CE1','NE2','HE2'],[HBOND_DIST,180,0],'don'],
                 ['HIP',['CE1','ND1','HD1'],[HBOND_DIST,180,0],'don'],
                 
                 ['HOH',['H1','O','H2'],[HBOND_DIST,180,0],'don'],
                 ['HOH',['H2','O','H1'],[HBOND_DIST,180,0],'don'],
                 ['WAT',['H1','O','H2'],[HBOND_DIST,180,0],'don'],
                 ['WAT',['H2','O','H1'],[HBOND_DIST,180,0],'don'],]

	lists = sc_acceptors + sc_donors
	
	return lists


# define ideal H bond interactions for protein residues (backbone)
def ideal_interactions_bb(HBOND_DIST):
	
	''' Returns a list of lists with ideal H bond interaction definitions:
		distance, angle, diherdal.
	'''

	# backbone interactions (carbonyl)
	bb_acceptors = [['XXX',['CA','C','O'],[HBOND_DIST,180,0],'acc'],
                    ['XXX',['CA','C','O'],[HBOND_DIST,210,0],'acc'],
                    ['XXX',['CA','C','O'],[HBOND_DIST,240,0],'acc'],
                    ['XXX',['CA','C','O'],[HBOND_DIST,150,0],'acc'],
                    ['XXX',['CA','C','O'],[HBOND_DIST,120,0],'acc']]

	# backbone interactions (NH) -Pro excluded-
	bb_donors = [['XXX',['CA','N','H'],[HBOND_DIST,180,0],'don']]
	
	lists = bb_acceptors + bb_donors
	
	return lists


# define aromatic interactions
def aromatic_interactions(AROM_DIST):
	
	''' Returns a list of lists with aromatic interactions
		Sintax: resname, ring atoms names, normal distance to centroid of ring,
		parallel distance (for parallel-displaced geometry)
	'''

	lists = [['PHE',['CG','CD1','CD2','CE1','CE2','CZ'],AROM_DIST,1.5],
             ['TYR',['CG','CD1','CD2','CE1','CE2','CZ'],AROM_DIST,1.5],
             ['TRP',['CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],AROM_DIST,1.5],
             ['HIS',['CG','ND1','CD2','CE1','NE2'],AROM_DIST,1.5],
             ['HIE',['CG','ND1','CD2','CE1','NE2'],AROM_DIST,1.5],
             ['HID',['CG','ND1','CD2','CE1','NE2'],AROM_DIST,1.5],
             ['HIP',['CG','ND1','CD2','CE1','NE2'],AROM_DIST,1.5]]

	return lists


# get centroid of coordinates
def centroid(vecs_array):
	'''
	Returns centroid of points
	array of 1x3 matrix with coordinates of points 
	'''
	xmean = np.mean(vecs_array[:,0])
	ymean = np.mean(vecs_array[:,1])
	zmean = np.mean(vecs_array[:,2])

	center = np.array([xmean,ymean,zmean])
	
	return center


# get normal vector to a plane
def normal_vector(a,b,c,magnitude):

	'''
	Returns normal vector with desired magnitude (as array)
	a,b,c are the vectors defining the plane (as arrays)
	cross product ca x cb
	'''

	# in-plane vectors
	d = a - c
	e = b - c

	# normal vector
	normal_temp = np.cross(d,e)

	# length of normal vector
	magnitude_temp = np.linalg.norm(normal_temp)

	# normal vector with desired magnitude
	normal = normal_temp / magnitude_temp * magnitude

	return normal


# handle rotations
def rotation_matrix(axis, theta):

	'''
	Return rotation matrix associated with counterclockwise
	rotation about the given axis by theta radians
	'''

	axis = np.asarray(axis)
	axis = axis/mt.sqrt(np.dot(axis, axis))
	a = mt.cos(theta/2.0)
	b, c, d = -axis * mt.sin(theta/2.0)
	aa, bb, cc, dd = a*a, b*b, c*c, d*d
	bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

	return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotate_around_axis(axis, theta, v):
	
	return np.dot(rotation_matrix(axis,theta), v)
    

# get dihedral angle between 4 points defining 2 planes
def get_dihedral(p1,p2,p3,p4):

    v1 = -1.0*(p2 - p1)
    v2 = p3 - p2
    v3 = p4 - p3

    # normalize v2 so that it does not influence magnitude of vector
    # rejections that come next
    v2 /= np.linalg.norm(v2)

    # vector rejections
    # v = projection of v1 onto plane perpendicular to v2
    #   = v1 minus component that aligns with v2
    # w = projection of v3 onto plane perpendicular to v2
    #   = v3 minus component that aligns with v2
    v = v1 - np.dot(v1, v2)*v2
    w = v3 - np.dot(v3, v2)*v2

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(v2, v), w)
    
    return np.degrees(np.arctan2(y, x))


# get 4th atom position given distance, angle and dihedral
def pos4(x1,y1,z1,x2,y2,z2,x3,y3,z3,r4,a4,d4):
	
	"""
	Returns x,y,z coordinates of point 4 satisfying:
	- distance between points 3-4 = r4
	- angle between points 2-3-4 = a4
	- dihedral between points 1-2-3-4 = d4
	"""
	
	xejx = (y3-y2)*(z1-z2) - (z3-z2)*(y1-y2)
	yejx = -1 * ((x3-x2)*(z1-z2) - (z3-z2)*(x1-x2))
	zejx = (x3-x2)*(y1-y2)-(y3-y2)*(x1-x2)
	
	rejx = mt.sqrt(mt.pow(xejx,2) + mt.pow(yejx,2) + mt.pow(zejx,2))
	
	l1 = xejx/rejx
	
	r23 = mt.sqrt(mt.pow(x3-x2,2) + mt.pow(y3-y2,2) + mt.pow(z3-z2,2))
	m1 = yejx/rejx
	n1 = zejx/rejx
	
	l2 = (x3-x2)/r23
	m2 = (y3-y2)/r23
	n2 = (z3-z2)/r23
	
	xejz = yejx*(z3-z2) - zejx*(y3-y2)
	yejz = -1 * (xejx*(z3-z2) - zejx*(x3-x2))
	zejz = xejx*(y3-y2) - yejx*(x3-x2)
	
	rejz = mt.sqrt(mt.pow(xejz,2) + mt.pow(yejz,2) + mt.pow(zejz,2))
	
	l3 = xejz/rejz
	m3 = yejz/rejz
	n3 = zejz/rejz
	
	d4 = d4*mt.pi/180
	a4 = 180-a4
	a4 = a4*mt.pi/180
	
	z = r4 * mt.sin(a4) * mt.cos(d4)
	x = r4 * mt.sin(a4) * mt.sin(d4)
	y = r4 * mt.cos(a4)
	
	y = y + r23
	
	x4 = l1*x + l2*y + l3*z + x2
	y4 = m1*x + m2*y + m3*z + y2
	z4 = n1*x + n2*y + n3*z + z2

	return(x4, y4, z4)


# get a line with PDB format
def write_PDB_format(resname,resid,name,atom_id,coordinates):
	x = coordinates[0]
	y = coordinates[1]
	z = coordinates[2]
	pdb_line = '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'.format('ATOM',atom_id,name,' ',resname,'X',resid,' ',x,y,z,1.00,0.00,'','')
	return(pdb_line)

# get a line with bpf file format
def write_bpf_format(htype,coordinates,vset=-1.5,r=2):
	x = coordinates[0]
	y = coordinates[1]
	z = coordinates[2]
	pdb_line = '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}   {:3s}'.format(x,y,z,vset,r,htype)
	return(pdb_line)



########## MAIN ##########


# save arguments to dictionary
arguments = optparser()

# parse the PDB file with biopython
parser = PDBParser(PERMISSIVE=1)
structure_id = 'protein'
filename = arguments['pdb_file']
structure = parser.get_structure(structure_id, filename)
# get the first model from the PDB file (there is usually only one in crystal structures)
model = structure[0]

# get the ideal HB interactions a protein could make
ideal_interactions = ideal_interactions(HBOND_DIST)
ideal_interactions_bb = ideal_interactions_bb(HBOND_DIST)


if 'dir' in arguments:
	dir = arguments['dir']
else:
	dir = '.'

# open a file to store the interaction sites
f = open(dir + '/interaction_sites.pdb','w')
# bias file
print(filename)
fb = open(dir + "/interaction_sites.bpf",'w')
fb.write("\tx\ty\tz\tVset\tr\ttype\n")


# get the interaction sites for the specified atoms
count = 1
for hb in arguments['hbs']:
	chainid = hb.split(':')[0]
	# get the chain	
	chain = model[chainid]

	# get residue
	resid = hb.split(':')[1]
	try:
		# raises error if it is water
		residue = chain[int(resid)]
	except:
		# for water residues
		residue = chain['W',int(resid),' ']
	# get residue name
	resname = residue.get_resname()

	# get atom type, i.e. name
	name = hb.split(':')[2]
	for atom_ in residue:
		if atom_.get_name() == name:
			atom = atom_
	print("Adding H-bond bias for",atom)
		
	# side chain HB interactions
	for interaction in ideal_interactions:
		
		# if atom is interactor according to ideal interactions previously defined
		if resname == interaction[0] and name == interaction[1][2]:
			# get coordinates of the three atoms defining the interaction
			coords1 = residue[interaction[1][0]].get_vector()
			coords2 = residue[interaction[1][1]].get_vector()
			coords3 = atom.get_vector()\

			# alcohols need special treatment to get lone pairs location as acceptors
			if (resname == 'SER' or resname == 'THR' or resname == 'TYR') and (interaction[2][2] == 'alpha' or interaction[2][2] == 'beta'):
				# get H atom coordinates
				if resname == 'SER':
					coords4 = residue['HG'].get_vector()
				elif resname == 'THR':
					coords4 = residue['HG1'].get_vector()
				elif resname == 'TYR':
					coords4 = residue['HH'].get_vector()
				# calculate C-C-O-H dihedral
				p1 = np.asarray([coords1[0],coords1[1],coords1[2]])
				p2 = np.asarray([coords2[0],coords2[1],coords2[2]])
				p3 = np.asarray([coords3[0],coords3[1],coords3[2]])
				p4 = np.asarray([coords4[0],coords4[1],coords4[2]])
				dihedral = get_dihedral(p1,p2,p3,p4)
				# calculate C-C-O-lone pair dihedrals and write donor interaction site in those positions
				if interaction[2][2] == 'alpha':
					ang1 = dihedral + 120
					# pos4 needs the coordinates of the three atoms defining the interaction + distance + angle + dihedral
					position = pos4(coords1[0],coords1[1],coords1[2],coords2[0],coords2[1],coords2[2],coords3[0],coords3[1],coords3[2],interaction[2][0],interaction[2][1],ang1)
					# write pdb line for interaction site to file
					pdb_line = write_PDB_format('DON',count,'H',count,position)
					bpf_line = write_bpf_format('don',position)
					count = count + 1
					f.write(pdb_line + '\n')
					fb.write(bpf_line + '\n')
				elif interaction[2][2] == 'beta':
					ang2 = dihedral + 240
					# pos4 needs the coordinates of the three atoms defining the interaction + distance + angle + dihedral
					position = pos4(coords1[0],coords1[1],coords1[2],coords2[0],coords2[1],coords2[2],coords3[0],coords3[1],coords3[2],interaction[2][0],interaction[2][1],ang2)
					# write pdb line for interaction site to file
					pdb_line = write_PDB_format('DON',count,'H',count,position)
					bpf_line = write_bpf_format('don',position)
					count = count + 1
					f.write(pdb_line + '\n')
					fb.write(bpf_line + '\n')
					
			# for non alcohol groups (and alcohols as donors)
			else:
				# if protein atom act as donor, write an acceptor site according to ideal interaction
				if interaction[3] == 'don':
					# pos4 needs the coordinates of the three atoms defining the interaction + distance + angle + dihedral
					position = pos4(coords1[0],coords1[1],coords1[2],coords2[0],coords2[1],coords2[2],coords3[0],coords3[1],coords3[2],interaction[2][0],interaction[2][1],interaction[2][2])
					# write pdb line for interaction site to file
					pdb_line = write_PDB_format('ACC',count,'O',count,position)
					bpf_line = write_bpf_format('acc',position)
					count = count + 1
					f.write(pdb_line + '\n')
					fb.write(bpf_line + '\n')
				# if protein atom act as acceptor, write a donor site according to ideal interaction
				elif interaction[3] == 'acc':
					# pos4 needs the coordinates of the three atoms defining the interaction + distance + angle + dihedral
					position = pos4(coords1[0],coords1[1],coords1[2],coords2[0],coords2[1],coords2[2],coords3[0],coords3[1],coords3[2],interaction[2][0],interaction[2][1],interaction[2][2])
					# write pdb line for interaction site to file
					pdb_line = write_PDB_format('DON',count,'H',count,position)
					bpf_line = write_bpf_format('don',position)
					count = count + 1
					f.write(pdb_line + '\n')
					fb.write(bpf_line + '\n')
		
	# backbone HB interactions
	if resname != 'HOH' and resname != 'WAT':

		for interaction in ideal_interactions_bb:
			# if atom is interactor according to ideal interactions previously defined
			if name == interaction[1][2]:
				# get coordinates of the three atoms defining the interaction
				coords1 = residue[interaction[1][0]].get_vector()
				coords2 = residue[interaction[1][1]].get_vector()
				coords3 = atom.get_vector()
				# pos4 needs the coordinates of the three atoms defining the interaction + distance + angle + dihedral
				position = pos4(coords1[0],coords1[1],coords1[2],coords2[0],coords2[1],coords2[2],coords3[0],coords3[1],coords3[2],interaction[2][0],interaction[2][1],interaction[2][2])
				# if interactor is carbonyl				
				# write a donor interactor at 5 different locations surrounding the carbonyl (ideal = 120 and 240 deg)
				if interaction[3] == 'acc':
					# write pdb line for interaction site to file
					pdb_line = write_PDB_format('DON',count,'H',count,position)
					bpf_line = write_bpf_format('don',position)
					count = count + 1
					f.write(pdb_line + '\n')
					fb.write(bpf_line + '\n')
				# if interactor is NH
				# write an acceptor interactor at HBOND_DIST angstroms from the H in the straight line defined by N-H 
				elif interaction[3] == 'don':
					# write pdb line for interaction site to file
					pdb_line = write_PDB_format('ACC',count,'O',count,position)
					bpf_line = write_bpf_format('acc',position)
					count = count + 1
					f.write(pdb_line + '\n')
					fb.write(bpf_line + '\n')

	# # aromatic interactions
	# for aro_int in aromatic_interactions(AROM_DIST):
	# 	# if residue is aromatic (Phe, Tyr, Trp, His)
	# 	if resname == aro_int[0]:
	# 		# get coordinates of ring atoms
	# 		coords_array = []
	# 		for ring_atom in aro_int[1]:
	# 			at_coords = residue[ring_atom].get_vector()
	# 			ar_coords = np.array([at_coords[0],at_coords[1],at_coords[2]])
	# 			coords_array.append(ar_coords)
	# 		coords_array = np.asarray(coords_array)
	# 		# get ring centroid
	# 		ring_center_coords = centroid(coords_array)
	# 		# get vector normal to the ring plane (magnitude defined in aromatic_interactions)
	# 		normal = normal_vector(coords_array[0],coords_array[1],coords_array[2],aro_int[2])
	# 		# get locations above and below the ring (stacked interaction)
	# 		above_ring = ring_center_coords + normal
	# 		below_ring = ring_center_coords - normal
	# 		# write stacked interaction sites to pdb file
	# 		for rel_pos in [above_ring, below_ring]:
	# 			pdb_line = write_PDB_format('ARO',count,'C',count,rel_pos)
	# 			count = count + 1
	# 			f.write(pdb_line + '\n')
	# 		# get vector contained in the ring plane (parallel)
	# 		dir1 = ring_center_coords
	# 		dir2 = coords_array[0]
	# 		parallel_vec = dir2 - dir1
	# 		parallel_vec_magnitude = np.linalg.norm(parallel_vec)
	# 		# displace stacked interaction sites in six senses according to parallel-displaced interaction
	# 		# define the 3 rotation angles
	# 		for rot_ang in [0,1.0472,2.0944]:
	# 			# rotate the parallel vector around normal axis
	# 			parallel_vec2 = rotate_around_axis(normal,rot_ang,parallel_vec)
	# 			# repeat for interactors both above and below the ring
	# 			for rel_pos in [above_ring, below_ring]:
	# 				# do the actual displacement and
	# 				# write parallel-displaced interaction sites to pdb file
	# 				parallel_displaced = rel_pos + parallel_vec2/parallel_vec_magnitude*aro_int[3]
	# 				pdb_line = write_PDB_format('ARO',count,'C',count,parallel_displaced)
	# 				count = count + 1
	# 				f.write(pdb_line + '\n')
	# 				parallel_displaced = rel_pos - parallel_vec2/parallel_vec_magnitude*aro_int[3]
	# 				pdb_line = write_PDB_format('ARO',count,'C',count,parallel_displaced)
	# 				count = count + 1
	# 				f.write(pdb_line + '\n')


