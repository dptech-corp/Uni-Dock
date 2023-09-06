from typing import List
import os
import math
import numpy as np
from Bio.PDB import PDBParser


# distance (A) between O/N---H in hydrogen bonds
HBOND_DIST = 1.9
# distance (A) between planes in aromatic interactions
AROM_DIST = 3.8

VSET = -1.5, 
RADIUS = 2.0


class HBondBias:
    def __init__(self, receptor_file:str, hbond_sites:str):
        self.receptor_file = receptor_file
        self.hbond_sites = hbond_sites
        self.ideal_interactions = self._get_ideal_interactions(HBOND_DIST)
        self.ideal_interactions_bb = self._get_ideal_interactions_bb(HBOND_DIST)
    
    def _get_ideal_interactions(self, hbond_dist:float):

        ''' Returns a list of lists with ideal H bond interaction definitions:
            distance, angle, diherdal.
        '''

        # side chain interactions (acceptors)
        # sintax: [resname,[atom1,atom2,atom3],[distance,angle,dihedral],type]
        sc_acceptors = [['GLU',['CG','CD','OE1'],[hbond_dist,180,0],'acc'],
                        ['GLU',['CG','CD','OE1'],[hbond_dist,210,0],'acc'],
                        ['GLU',['CG','CD','OE1'],[hbond_dist,240,0],'acc'],
                        ['GLU',['CG','CD','OE1'],[hbond_dist,150,0],'acc'],
                        ['GLU',['CG','CD','OE1'],[hbond_dist,120,0],'acc'],

                        ['GLU',['CG','CD','OE2'],[hbond_dist,180,0],'acc'],
                        ['GLU',['CG','CD','OE2'],[hbond_dist,210,0],'acc'],
                        ['GLU',['CG','CD','OE2'],[hbond_dist,240,0],'acc'],
                        ['GLU',['CG','CD','OE2'],[hbond_dist,150,0],'acc'],
                        ['GLU',['CG','CD','OE2'],[hbond_dist,120,0],'acc'],

                        ['ASP',['CB','CG','OD1'],[hbond_dist,180,0],'acc'],
                        ['ASP',['CB','CG','OD1'],[hbond_dist,210,0],'acc'],
                        ['ASP',['CB','CG','OD1'],[hbond_dist,240,0],'acc'],
                        ['ASP',['CB','CG','OD1'],[hbond_dist,150,0],'acc'],
                        ['ASP',['CB','CG','OD1'],[hbond_dist,120,0],'acc'],

                        ['ASP',['CB','CG','OD2'],[hbond_dist,180,0],'acc'],
                        ['ASP',['CB','CG','OD2'],[hbond_dist,210,0],'acc'],
                        ['ASP',['CB','CG','OD2'],[hbond_dist,240,0],'acc'],
                        ['ASP',['CB','CG','OD2'],[hbond_dist,150,0],'acc'],
                        ['ASP',['CB','CG','OD2'],[hbond_dist,120,0],'acc'],

                        ['ASN',['CB','CG','OD1'],[hbond_dist,180,0],'acc'],
                        ['ASN',['CB','CG','OD1'],[hbond_dist,210,0],'acc'],
                        ['ASN',['CB','CG','OD1'],[hbond_dist,240,0],'acc'],
                        ['ASN',['CB','CG','OD1'],[hbond_dist,150,0],'acc'],
                        ['ASN',['CB','CG','OD1'],[hbond_dist,120,0],'acc'],

                        ['GLN',['CG','CD','OE1'],[hbond_dist,180,0],'acc'],
                        ['GLN',['CG','CD','OE1'],[hbond_dist,210,0],'acc'],
                        ['GLN',['CG','CD','OE1'],[hbond_dist,240,0],'acc'],
                        ['GLN',['CG','CD','OE1'],[hbond_dist,150,0],'acc'],
                        ['GLN',['CG','CD','OE1'],[hbond_dist,120,0],'acc'],

                        ['SER',['CA','CB','OG'], [hbond_dist,109.5,'alpha'],'acc'],
                        ['SER',['CA','CB','OG'], [hbond_dist,109.5,'beta'], 'acc'],

                        ['THR',['CA','CB','OG1'],[hbond_dist,109.5,'alpha'],'acc'],
                        ['THR',['CA','CB','OG1'],[hbond_dist,109.5,'beta'], 'acc'],

                        ['TYR',['CE1','CZ','OH'],[hbond_dist,109.5,'alpha'],'acc'],
                        ['TYR',['CE1','CZ','OH'],[hbond_dist,109.5,'beta'], 'acc'],

                        ['HID',['CG','CD2','NE2'], [hbond_dist,125.4,180],'acc'],
                        ['HIE',['NE2','CE1','ND1'],[hbond_dist,125.4,180],'acc'],
                        
                        ['HOH',['H1','H2','O'],    [hbond_dist,109.5,120],'acc'],
                        ['HOH',['H1','H2','O'],    [hbond_dist,109.5,240],'acc'],
                        ['WAT',['H1','H2','O'],    [hbond_dist,109.5,120],'acc'],
                        ['WAT',['H1','H2','O'],    [hbond_dist,109.5,240],'acc']]

        # side chain interactions (donors)
        # sintax: [resname,[atom1,atom2,atom3],[distance,angle,dihedral],type]
        sc_donors = [['ASN',['CG','ND2','HD21'],[hbond_dist,180,0],'don'],
                    ['ASN',['CG','ND2','HD22'],[hbond_dist,180,0],'don'],

                    ['GLN',['CD','NE2','HE21'],[hbond_dist,180,0],'don'],
                    ['GLN',['CD','NE2','HE22'],[hbond_dist,180,0],'don'],

                    ['ARG',['CZ','NE','HNE'],  [hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NH1','HH11'],[hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NH1','HH12'],[hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NH2','HH21'],[hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NH2','HH22'],[hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NE','HE'],   [hbond_dist,180,0],'don'],

                    ['SER',['CB','OG','HG'],[hbond_dist,180,0],'don'],

                    ['THR',['CB','OG1','HG1'],[hbond_dist,180,0],'don'],

                    ['TYR',['CZ','OH','HH'],[hbond_dist,180,0],'don'],

                    ['TRP',['CD1','NE1','HE1'],[hbond_dist,180,0],'don'],

                    ['LYS',['CE','NZ','HZ1'],[hbond_dist,180,0],'don'],
                    ['LYS',['CE','NZ','HZ2'],[hbond_dist,180,0],'don'],
                    ['LYS',['CE','NZ','HZ3'],[hbond_dist,180,0],'don'],

                    ['HIE',['CE1','NE2','HE2'],[hbond_dist,180,0],'don'],
                    ['HID',['CE1','ND1','HD1'],[hbond_dist,180,0],'don'],
                    ['HIP',['CE1','NE2','HE2'],[hbond_dist,180,0],'don'],
                    ['HIP',['CE1','ND1','HD1'],[hbond_dist,180,0],'don'],
                    
                    ['HOH',['H1','O','H2'],[hbond_dist,180,0],'don'],
                    ['HOH',['H2','O','H1'],[hbond_dist,180,0],'don'],
                    ['WAT',['H1','O','H2'],[hbond_dist,180,0],'don'],
                    ['WAT',['H2','O','H1'],[hbond_dist,180,0],'don'],]

        lists = sc_acceptors + sc_donors
        
        return lists

    # define ideal H bond interactions for protein residues (backbone)
    def _get_ideal_interactions_bb(self, hbond_dist:float):
    
        ''' Returns a list of lists with ideal H bond interaction definitions:
            distance, angle, diherdal.
        '''

        # backbone interactions (carbonyl)
        bb_acceptors = [['XXX',['CA','C','O'],[hbond_dist,180,0],'acc'],
                        ['XXX',['CA','C','O'],[hbond_dist,210,0],'acc'],
                        ['XXX',['CA','C','O'],[hbond_dist,240,0],'acc'],
                        ['XXX',['CA','C','O'],[hbond_dist,150,0],'acc'],
                        ['XXX',['CA','C','O'],[hbond_dist,120,0],'acc']]

        # backbone interactions (NH) -Pro excluded-
        bb_donors = [['XXX',['CA','N','H'],[hbond_dist,180,0],'don']]
        
        lists = bb_acceptors + bb_donors
        
        return lists

    # get dihedral angle between 4 points defining 2 planes
    def get_dihedral(self, p1, p2, p3, p4):

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
    def pos4(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, r4, a4, d4):
        
        """
        Returns x,y,z coordinates of point 4 satisfying:
        - distance between points 3-4 = r4
        - angle between points 2-3-4 = a4
        - dihedral between points 1-2-3-4 = d4
        """
        
        xejx = (y3-y2)*(z1-z2) - (z3-z2)*(y1-y2)
        yejx = -1 * ((x3-x2)*(z1-z2) - (z3-z2)*(x1-x2))
        zejx = (x3-x2)*(y1-y2)-(y3-y2)*(x1-x2)
        
        rejx = math.sqrt(math.pow(xejx,2) + math.pow(yejx,2) + math.pow(zejx,2))
        
        l1 = xejx/rejx
        
        r23 = math.sqrt(math.pow(x3-x2,2) + math.pow(y3-y2,2) + math.pow(z3-z2,2))
        m1 = yejx/rejx
        n1 = zejx/rejx
        
        l2 = (x3-x2)/r23
        m2 = (y3-y2)/r23
        n2 = (z3-z2)/r23
        
        xejz = yejx*(z3-z2) - zejx*(y3-y2)
        yejz = -1 * (xejx*(z3-z2) - zejx*(x3-x2))
        zejz = xejx*(y3-y2) - yejx*(x3-x2)
        
        rejz = math.sqrt(math.pow(xejz,2) + math.pow(yejz,2) + math.pow(zejz,2))
        
        l3 = xejz/rejz
        m3 = yejz/rejz
        n3 = zejz/rejz
        
        d4 = d4*math.pi/180
        a4 = 180-a4
        a4 = a4*math.pi/180
        
        z = r4 * math.sin(a4) * math.cos(d4)
        x = r4 * math.sin(a4) * math.sin(d4)
        y = r4 * math.cos(a4)
        
        y = y + r23
        
        x4 = l1*x + l2*y + l3*z + x2
        y4 = m1*x + m2*y + m3*z + y2
        z4 = n1*x + n2*y + n3*z + z2

        return (x4, y4, z4)

    # get a line with bpf file format
    def write_bpf_format(self, htype:str, coordinates:List[float], vset:float=VSET, r:float=RADIUS):
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        bpf_line = f'{x:6.3f} {y:6.3f} {z:6.3f} {vset:6.2f} {r:6.2f} {htype:3s}'
        return bpf_line
 
    def gen_hbond_bias(self, out_bias_file:str):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure("protein", self.receptor_file)
        model = structure[0]

        os.makedirs(os.path.dirname(os.path.abspath(out_bias_file)), exist_ok=True)

        bpf_lines = list()
        count = 1
        for hb in self.hbond_sites.split(","):
            chainid, resid, atomname = hb.split(":")
            # get the chain	
            chain = model[chainid]
            try:
                # raises error if it is water
                insertion_code = ' '
                if resid[-1].isalpha():
                    insertion_code = resid[-1]     
                    resid = resid[:-1]   
                residue = chain[' ', int(resid), insertion_code]
            except:
                # for water residues
                residue = chain['W', int(resid), ' ']
            # get residue name
            resname = residue.get_resname()

            for atom_ in residue:
                if atom_.get_name() == atomname:
                    atom = atom_
                    break
                
            # side chain HB interactions
            for interaction in self.ideal_interactions:    
                # if atom is interactor according to ideal interactions previously defined
                if resname == interaction[0] and atomname == interaction[1][2]:
                    # get coordinates of the three atoms defining the interaction
                    coords1 = residue[interaction[1][0]].get_vector()
                    coords2 = residue[interaction[1][1]].get_vector()
                    coords3 = atom.get_vector()

                    # alcohols need special treatment to get lone pairs location as acceptors
                    if resname in ['SER', 'THR', 'TYR'] and interaction[2][2] in ['alpha', 'beta']:
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
                        dihedral = self.get_dihedral(p1, p2, p3, p4)
                        if interaction[2][2] == 'alpha':
                            ang = dihedral + 120
                        elif interaction[2][2] == 'beta':
                            ang = dihedral + 240
                        position = self.pos4(coords1[0], coords1[1], coords1[2], 
                            coords2[0], coords2[1], coords2[2], coords3[0], 
                            coords3[1], coords3[2], interaction[2][0], 
                            interaction[2][1], ang)
                        bpf_line = self.write_bpf_format('don', position)
                        count += 1
                        bpf_lines.append(bpf_line)
                            
                    # for non alcohol groups (and alcohols as donors)
                    else:
                        if interaction[3] in ['don', 'acc']:
                            # if protein atom act as donor, write an acceptor site according to ideal interaction
                            position = self.pos4(coords1[0], coords1[1], coords1[2], 
                                coords2[0], coords2[1], coords2[2], coords3[0], 
                                coords3[1], coords3[2], interaction[2][0], 
                                interaction[2][1], interaction[2][2])
                            if interaction[3] == 'don':
                                bpf_line = self.write_bpf_format('acc', position)
                            # if protein atom act as acceptor, write a donor site according to ideal interaction
                            elif interaction[3] == 'acc':
                                bpf_line = self.write_bpf_format('don', position)
                            count += 1
                            bpf_lines.append(bpf_line)
                
            # backbone HB interactions
            if resname != 'HOH' and resname != 'WAT':
                for interaction in self.ideal_interactions_bb:
                    # if atom is interactor according to ideal interactions previously defined
                    if atomname == interaction[1][2]:
                        # get coordinates of the three atoms defining the interaction
                        coords1 = residue[interaction[1][0]].get_vector()
                        coords2 = residue[interaction[1][1]].get_vector()
                        coords3 = atom.get_vector()
                        # pos4 needs the coordinates of the three atoms defining the interaction + distance + angle + dihedral
                        position = self.pos4(coords1[0], coords1[1], coords1[2], 
                            coords2[0], coords2[1], coords2[2], coords3[0], 
                            coords3[1], coords3[2], interaction[2][0], 
                            interaction[2][1], interaction[2][2])
                        # if interactor is carbonyl				
                        # write a donor interactor at 5 different locations surrounding the carbonyl (ideal = 120 and 240 deg)
                        if interaction[3] in ['acc', 'don']:
                            if interaction[3] == 'acc':
                                bpf_line = self.write_bpf_format('don', position)
                            # if interactor is NH
                            # write an acceptor interactor at HBOND_DIST angstroms from the H in the straight line defined by N-H 
                            elif interaction[3] == 'don':
                                bpf_line = self.write_bpf_format('acc', position)
                            count += 1
                            bpf_lines.append(bpf_line)
        
        with open(out_bias_file, 'w') as f:
            f.write("\tx\ty\tz\tVset\tr\ttype\n")
            for line in bpf_lines:
                f.write(line + '\n')