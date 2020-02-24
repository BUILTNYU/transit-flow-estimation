
def open_data_file_with_header(file_dir):
    """Input file dir; output a list with header removed and elmt decomposed."""

    with open(file_dir,'rU') as file:
        file_lines = file.readlines()
        
        # note: there is a \n at the end of each line
        file_lines = [item.rstrip() for item in file_lines]

        # delete the header
        del file_lines[0]

        # decompose
        file_list = [item.split(',') for item in file_lines] 

    return file_list