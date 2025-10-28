

def obtain_ridx_material_info(material: str) -> tuple:
    """
    Obtain the shelf, book, and page information for a given material from the RefractiveIndex database.

    Parameters
    ----------
    material : str
        The name of the material.

    Returns
    -------
    tuple
        A tuple containing the shelf, book, and page information.
    """
    material_info = {
        "Au": ("main", "Au", "Babar"),
        # Add more materials as needed
    }

    if material in material_info:
        return material_info[material]
    else:
        raise ValueError(f"Material '{material}' not found in the database.")