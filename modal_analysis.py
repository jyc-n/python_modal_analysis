'''
  A python modal analysis tool for the reduced deformable simulation
'''
import argparse
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

class Mesh:

  def __init__(self):
    self.nodes = np.empty((0, 3), dtype=np.double)
    self.tets = np.empty((0, 4), dtype=np.int32)
    self.num_nodes = 0
    self.num_tets = 0

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

  def load_from_obj(self, filename):
    '''Read mesh from an obj file.'''
    print('called obj: ' + filename)

'''Modal analyzer'''
class ModalAnalyzer:
  _stiffness_matrix: csr_matrix
  _mass_matrix: csr_matrix

  def __init__(self, mesh):
    self._mesh = mesh
    self._youngs_modulus = 1.0e5
    self._rho = 1.0e3
    self._area = 1.0e-4

  def _get_index(self, node_1, node_2):
    '''Get the unique index of the edge.'''
    cantor_pair = lambda i, j: int((i + j) * (i + j + 1) / 2 + j)
    return cantor_pair(node_1, node_2) if node_1 > node_2 else cantor_pair(node_2, node_1)

  def _build_mass_spring_model(self):
    '''Internal method to build matrices for the mass-spring model.'''
    print('Building mass-spring model...')

    node_order = np.array([[0, 1], [1, 2], [2, 0], [3, 0], [3, 1], [3, 2]], dtype=int)
    edge_lookup_table = {}

    for i_tet in range(self._mesh.num_tets):
      for i in range(6):
        node_1 = self._mesh.tets[i_tet, node_order[i, 0]]
        node_2 = self._mesh.tets[i_tet, node_order[i, 1]]
        edge_index = self._get_index(node_1, node_2)

        if edge_index not in edge_lookup_table:
          edge_lookup_table[edge_index] = [node_1, node_2]

    # auxiliary matrices for building the sparse matrix
    global_matrix_size = self._mesh.num_nodes * 3
    k_buffer = dok_matrix((global_matrix_size, global_matrix_size), dtype=np.double)
    m_buffer = dok_matrix((global_matrix_size, global_matrix_size), dtype=np.double)

    for edge in edge_lookup_table.values():
      # global node index
      node_1 = edge[0]
      node_2 = edge[1]
      # nodal coordinates
      x_1 = self._mesh.nodes[node_1, :]
      x_2 = self._mesh.nodes[node_2, :]

      # transformation matrix
      length = np.linalg.norm(x_1 - x_2, ord=2)
      c_x = (x_1[0] - x_2[0]) / length
      c_y = (x_1[1] - x_2[1]) / length
      c_z = (x_1[2] - x_2[2]) / length
      t_matrix = np.array([[c_x, c_y, c_z, 0, 0, 0],
                           [0, 0, 0, c_x, c_y, c_z]])

      # local stiffness matrix
      k_s = self._youngs_modulus * self._area / length
      k_local = t_matrix.transpose() @ np.array([[k_s, -k_s], [-k_s, k_s]]) @ t_matrix

      # local mass matrix
      m_local = np.identity(6, dtype=float) * self._rho * self._area * length * 0.5

      # build global matrix
      entry_1 = 3 * node_1
      entry_2 = 3 * node_2

      k_buffer[entry_1:entry_1+3, entry_1:entry_1+3] += k_local[:3, :3]
      k_buffer[entry_1:entry_1+3, entry_2:entry_2+3] += k_local[:3, 3:]
      k_buffer[entry_2:entry_2+3, entry_1:entry_1+3] += k_local[3:, :3]
      k_buffer[entry_2:entry_2+3, entry_2:entry_2+3] += k_local[3:, 3:]

      m_buffer[entry_1:entry_1+3, entry_1:entry_1+3] += m_local[:3, :3]
      m_buffer[entry_1:entry_1+3, entry_2:entry_2+3] += m_local[:3, 3:]
      m_buffer[entry_2:entry_2+3, entry_1:entry_1+3] += m_local[3:, :3]
      m_buffer[entry_2:entry_2+3, entry_2:entry_2+3] += m_local[3:, 3:]

    # converge to CSR matrix
    self._stiffness_matrix = k_buffer.tocsr()
    self._mass_matrix = m_buffer.tocsr()

    # enforce symmetry
    self._stiffness_matrix = 0.5 * (self._stiffness_matrix + self._stiffness_matrix.transpose())

    np.savetxt('stiffness.txt', k_buffer.todense(), fmt='%.16e')
    np.savetxt('mass.txt', m_buffer.todense(), fmt='%.16e')
    print('Mass-spring model building complete')


  def _build_linear_fem_model(self):
    '''Internal method to build matrices for the linearelastic FEM model.'''
    print('Building Linearelastic FEM model...')

    print('Linearelastic FEM model building complete')


  def build_matrice(self, cons_model):
    '''Build full-space matrices stiffness and mass matrices for the given mesh.'''

    if cons_model == 0:
      self._build_mass_spring_model()

    elif cons_model == 1:
      self._build_linear_fem_model()

    else:
      raise RuntimeError('Invalid cons_model arguments.')

  def eigen_solve(self, num_modes):
    '''Solve the generalized eigenvalue problems.'''

    num_modes = num_modes if num_modes > 0 else self._mesh.num_nodes * 3
    eigenvalues, eigenvectors = eigsh(self._stiffness_matrix, M=self._mass_matrix,
                                      k=num_modes, which='SM')
                                      # maxiter = 1000, tol=0)
    print('First 10 eigenvalues:')
    print(eigenvalues[:10])
    # print(eigenvectors)

  def compute_reduced_matrices(self):
    '''Compute the reduced mass and stiffness matrices'''



'''Modal analysis driver class'''
class AnalysisDriver:

  def __init__(self):
    self._mesh = Mesh()
    self._analyzer = ModalAnalyzer(self._mesh)

  def load_mesh(self, filename):
    '''Load mesh infomation from .vtk or .obj files for modal analysis. '''

    print('\nLoading: ' + filename)

    file_extension = filename.split('.')[-1]
    if file_extension == 'obj':
      self._mesh.load_from_obj(filename)

    elif file_extension == 'vtk':
      self._mesh.load_from_vtk(filename)

    else:
      raise RuntimeError('Unsupported file type: ' + file_extension +
                         '. Please check your input arguments.')

    print('Successfully read: ' + filename)
    print(f'#nodes: {self._mesh.num_nodes}')
    print(f'#tets: {self._mesh.num_tets}\n')

  def analyze(self, cons_model, num_modes):
    '''Run modal analysis for the given mesh.

    Args:
        cons_model: option for constitutive model.
                    0 is mass-spring, and 1 is linearelastic FEM.
        num_modes: the number of modes needs to be computed
    '''
    # build full space matrices K and M
    self._analyzer.build_matrice(cons_model)
    # solve the generalize eigenvalue problem
    self._analyzer.eigen_solve(num_modes)
    # compute the reduced mass and stiffness matrices
    self._analyzer.compute_reduced_matrices()

  def write_reduced_modes(self, output_dir):
    '''Write the reduced deformable files.

    This function writes the reduced stiffness matrix, nodal mass array,
    modes and eigenvalues to files. All output files are .bin files.

    Args:
        output_dir: directory to save all the output files
    '''
    # TODO

  def debug_reduced_mode_shape(self, mode, output_dir):
    '''Output the deformed mode shape for the given mode'''


def main():
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mesh', action='store', type=str,
                        help='name of the input mesh', required=True)
    parser.add_argument('--output_dir', action='store', type=str, default='',
                        help='output directory of reduced deformable files')
    parser.add_argument('--num_modes', action='store', type=int, default=-1,
                        help='Number of modes to compute')
    parser.add_argument('--cons_model', action='store', type=int, default=0,
                        help='Constitutive model: 0 is mass-spring, 1 is linear FEM')
    parser.add_argument('--debug_mode', action='store', type=int, default=-1,
                        help='check the mode shape of the given mode')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbose profiling')
    args = parser.parse_args()

    driver = AnalysisDriver()
    # load mesh from file
    driver.load_mesh(args.input_mesh)
    # run modal analysis
    driver.analyze(args.cons_model, args.num_modes)

  except RuntimeError as err:
    print(err.args[0])


if __name__ == '__main__':
  main()
