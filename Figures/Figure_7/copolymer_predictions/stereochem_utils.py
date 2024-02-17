from rdkit.Chem import AllChem

def get_single_hydroxyacid_chiral_smiles(smiles):
    # Return chiral smiles for a hydroxyacid
    # Define SMARTS
    mol = AllChem.MolFromSmiles(smiles)
    acid_smarts = AllChem.MolFromSmarts("[#8][#6]=[#8]")
    ol_smarts = AllChem.MolFromSmarts("[#8][#6][!#8]")

    # Find Atom Positions
    acid_C = mol.GetSubstructMatch(acid_smarts)[1]
    ol_C =  mol.GetSubstructMatch(ol_smarts)[0]

    # Find Shortest Path
    shortest_path = AllChem.GetShortestPath(mol, acid_C, ol_C)

    # Find Chiral Centers
    chiral_centers = AllChem.FindMolChiralCenters(mol, includeUnassigned=True)

    if len(chiral_centers) > 1:
        return "Error: Mulitple Sites Found"

    chiral_smiles = []
    for chiral_center in chiral_centers:
        atom = mol.GetAtoms()[chiral_center[0]]
        atom.SetChiralTag(AllChem.ChiralType.CHI_TETRAHEDRAL_CW)
        AllChem.AssignCIPLabels(mol)

        mol_smiles = AllChem.MolToSmiles(mol)
        if "@@" in mol_smiles:
            mol_smiles = ".".join([mol_smiles, mol_smiles.replace("@@", "@")])
        else:
            mol_smiles = ".".join([mol_smiles, mol_smiles.replace("@", "@@")])

        chiral_smiles.append(mol_smiles)
        mol = AllChem.MolFromSmiles(AllChem.MolToSmiles(mol).replace("@", ""))
    
    return chiral_smiles[0]

def add_stereo_smiles_to_df(df_monomers, return_valid_only=True):
    df_monomers = df_monomers.copy()
    for i, row in df_monomers.iterrows():
        try:
            df_monomers.loc[i, "smiles_stereo"] = get_single_hydroxyacid_chiral_smiles(row.smiles_monomer)
        except BaseException as e:
            df_monomers.loc[i, "smiles_stereo"] = "Error: Unable to Generate Chiral Structure"
    
    if return_valid_only:
        return df_monomers[~df_monomers.smiles_stereo.str.startswith("Error")]
    
    return df_monomers

def get_stereoester_df(df_monomers, return_valid_only):
    df_monomers = add_stereo_smiles_to_df(df_monomers, return_valid_only=return_valid_only)
    df_monomers = df_monomers[~df_monomers.smiles_monomer.duplicated()]
    df_monomers = df_monomers.reset_index(drop=True)
    return df_monomers