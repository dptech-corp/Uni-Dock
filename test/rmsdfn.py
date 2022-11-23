from biopandas.pdb import PandasPdb
import pandas as pd
import numpy as np
import warnings

def get_coor_from_pdb(ligfile: str) -> dict:
    '''
    Get coordinates of the atoms in the ligand from a pdb file.
    
    Args:
      ligfile (str): the pdb file of the ligand
    
    Returns:
      A list of coordinates of atoms in the ligand.
    '''
    assert ligfile.endswith(".pdb"), \
        "The ligand file must be a pdb file."
    ppdb = PandasPdb().read_pdb(ligfile)
    df = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]])
    coor = []
    for i in range(len(df)):
        # if df.loc[i, "atom_name"] == "H":
        #     continue
        try:
            coor.append([
                df.loc[i, "x_coord"].item(), 
                df.loc[i, "y_coord"].item(), 
                df.loc[i, "z_coord"].item()
            ])
        except:
            warnings.warn("something wrong about atom %d"%i)
    return coor


def get_coor_from_pdbqt(ligfile: str) -> dict:
    '''
    Get coordinates of the atoms in the ligand from a pdb file.
    
    Args:
      ligfile (str): the pdbqt file of the ligand
    
    Returns:
      A list of coordinates of atoms in the ligand.
    '''
    with open(ligfile, "r") as f:
        lines = f.readlines()
    coor = []
    for line in lines:
        # print(line)
        # if df.loc[i, "atom_name"] == "H":
        #     continue
        if len(line) > 60 and line[0:4]=='ATOM':
            # print(line)
            if line[13] == 'H':
                continue
            coor.append([
                float(line[30:38].strip()), 
                float(line[38:46].strip()), 
                float(line[46:54].strip())
            ])
        if line[:7] == 'MODEL 2':
            break
    # print(coor)
    return coor

def _calc_rmsd(coor_lig1: list, coor_lig2: list) -> float:
    '''
    Calculate the root mean square deviation between two ligands.
    
    Args:
      coor_lig1 (list): list of coordinates of the first ligand
      coor_lig2 (list): list of coordinates of the second ligand
    
    Returns:
      The RMSD value.
    '''
    coor_lig1 = np.array(coor_lig1)
    coor_lig2 = np.array(coor_lig2)
    distance = np.sqrt(((coor_lig1 - coor_lig2)**2).sum(-1))
    return np.sqrt(np.mean(distance**2))


def rmsd(ligfile1: str, ligfile2: str) -> float:
    '''
    Calculate the RMSD between two ligands.
    
    Args:
      ligfile1 (str): the pdb file of the first ligand
      ligfile2 (str): the pdb file of the second ligand
    
    Returns:
      The RMSD value.
    '''
    coor_lig1 = get_coor_from_pdb(ligfile1)
    # print(coor_lig1)
    coor_lig2 = get_coor_from_pdb(ligfile2)
    # print(coor_lig2)
    return _calc_rmsd(coor_lig1, coor_lig2)

def fast_rmsd(ligfile1: str, ligfile2: str) -> float:
    '''
    Calculate the RMSD between two ligands.
    
    Args:
      ligfile1 (str): the pdb file of the first ligand
      ligfile2 (str): the pdb file of the second ligand
    
    Returns:
      The RMSD value.
    '''
    coor_lig1 = get_coor_from_pdbqt(ligfile1)
    # print(coor_lig1)
    coor_lig2 = get_coor_from_pdbqt(ligfile2)
    # print(coor_lig2)
    return _calc_rmsd(coor_lig1, coor_lig2)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Calculate the RMSD between two ligands.")
    parser.add_argument("-l1", "--ligfile1", required=True,
        help="the pdb file of the first ligand")
    parser.add_argument("-l2", "--ligfile2", required=True,
        help="the pdb file of the second ligand")
    args = parser.parse_args()

    print("RMSD between %s %s: %.3f"%(
        args.ligfile1, args.ligfile2, 
        rmsd(args.ligfile1, args.ligfile2)
    ))

if __name__ == "__main__":
    main()
