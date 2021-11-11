'''
  A python modal analysis tool for the reduced deformable simulation
'''
import argparse
import numpy as np
import scipy

class Mesh:
  nodes = np.empty((0, 3), dtype=np.double)
  tets = np.empty((0, 4), dtype=np.int32)
  num_nodes = 0
  num_tets = 0

  def load_from_vtk(self, filename):
    '''Read mesh from a vtk file.'''

    with open(filename, 'r', encoding='utf-8') as input_mesh:
      reading_points = False
      reading_tets = False
      aux_counter = 0
      nodes_list = []
      tets_list = []

      for line in input_mesh:
        current_line = line.split(' ')

        if current_line[0] == 'POINTS':
          self.num_nodes = int(current_line[1])
          reading_points = True
          aux_counter = self.num_nodes

        elif current_line[0] == 'CELLS':
          self.num_tets = int(current_line[1])
          reading_tets = True
          aux_counter = self.num_tets

        elif reading_points:
          current_node = [float(line.split(' ')[0]),
                          float(line.split(' ')[1]),
                          float(line.split(' ')[2])]
          nodes_list.append(current_node)

          aux_counter -= 1
          reading_points = aux_counter > 0

        elif reading_tets:
          current_tet = [int(line.split(' ')[1]),
                         int(line.split(' ')[2]),
                         int(line.split(' ')[3]),
                         int(line.split(' ')[4])]
          tets_list.append(current_tet)

          aux_counter -= 1
          reading_tets = aux_counter > 0

    self.nodes = np.asarray(nodes_list)
    self.tets = np.asarray(tets_list)
    print('Successfully read: ' + filename)
    print('#nodes: {self.nodes}')
    print('#tets: {self.tets}')

  def load_from_obj(self, filename):
    '''Read mesh from an obj file.'''
    print('called obj: ' + filename)


'''Modal analysis driver class

'''
class ModalAnalyzer:

  _num_modes = 0
  _mesh = Mesh()

  def load_mesh(self, filename):
    '''Load mesh infomation from .vtk or .obj files for modal analysis. '''
    print('filename: ' + filename)

    file_extension = filename.split('.')[-1]
    if file_extension == 'obj':
      self._mesh.load_from_obj(filename)

    elif file_extension == 'vtk':
      self._mesh.load_from_vtk(filename)

    else:
      raise RuntimeError('Unsupported file type: ' + file_extension +
                         '. Please check your input arguments.')

  def analyze(self):
    pass

  def write_reduced_modes(self):
    pass

  def debug_reduced_mode_shape(self):
    pass


def main():
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mesh', action='store', type=str,
                        help='name of the input mesh', required=True)
    parser.add_argument('--output_dir', action='store', type=str, default='',
                        help='output directory of reduced deformable files')
    parser.add_argument('--debug_mode', action='store', type=int, default=-1,
                        help='check the mode shape of the given mode')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbose profiling')
    args = parser.parse_args()

    analyzer = ModalAnalyzer()
    analyzer.load_mesh(args.input_mesh)

  except RuntimeError as err:
    print(err.args[0])


if __name__ == '__main__':
  main()
