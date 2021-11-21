'''
  A python modal analysis tool for the reduced deformable simulation
'''
import argparse
import numpy as np
from numpy.core.fromnumeric import diagonal
import scipy.linalg
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

'''Mesh class holds the vertices and connectivity info.'''
class Mesh:

  def __init__(self):
    self.nodes = np.empty((0, 3), dtype=np.double)
    self.nodes_original = np.empty((0, 3), dtype=np.double)
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
    self.nodes_original = self.nodes.copy()
    self.tets = np.asarray(tets_list)

  def load_from_obj(self, filename):
    '''Read mesh from an obj file.'''
    print('called obj: ' + filename)

  def write_obj(self, coords, filename):
    '''Write obj file using the provided coordinates.

    Args:
        coords: given coordinates of the current mesh.
        filename: name of the .obj file.
    '''

    face_order = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]], dtype=int)

    with open(filename, 'w', encoding='utf-8') as outfile:

      counter = 0
      for vertex in coords:
        outfile.write(f'v {vertex[0]:10.8f} {vertex[1]:10.8f} {vertex[2]:10.8f} 0 0 0\n')
        counter += 1
      outfile.write(f'# {counter:d} vertices\n')

      face_lookup = set()
      counter = 0
      for tet in self.tets:
        for f in range(4):
          face = tuple(sorted([tet[face_order[f, 0]],
                               tet[face_order[f, 1]],
                               tet[face_order[f, 2]]]))

          if face not in face_lookup:
            face_lookup.add(face)
            outfile.write(f'f {face[0]+1:d} {face[1]+1:d} {face[2]+1:d}\n')
            counter += 1

      outfile.write(f'# {counter:d} faces\n')


  def scale_mesh_to_solve(self, scale):
    '''Scale the so that it's easier to solve the eigenvalue problem.

    Args:
        scale: user-specified scaling factor for the mesh.
    '''

    dim_before = [np.amax(self.nodes[:, 0]) - np.amin(self.nodes[:, 0]),
                  np.amax(self.nodes[:, 1]) - np.amin(self.nodes[:, 1]),
                  np.amax(self.nodes[:, 2]) - np.amin(self.nodes[:, 2])]

    print(f'Dimensions before scaling: [{dim_before[0]}, {dim_before[1]}, {dim_before[2]}]')

    self.nodes[:, 0] *= scale
    self.nodes[:, 1] *= scale
    self.nodes[:, 2] *= scale

    dim_after = [np.amax(self.nodes[:, 0]) - np.amin(self.nodes[:, 0]),
                 np.amax(self.nodes[:, 1]) - np.amin(self.nodes[:, 1]),
                 np.amax(self.nodes[:, 2]) - np.amin(self.nodes[:, 2])]

    print(f'Dimensions after scaling: [{dim_after[0]}, {dim_after[1]}, {dim_after[2]}]')


'''Build the discretized model and solve the generalized eigenvalue problem.'''
class ModalAnalyzer:


  def __init__(self, mesh):
    self._mesh = mesh
    self._youngs_modulus = 1.0e5
    self._rho = 1.0e3
    self._area = 1.0e-4
    self._full_size = 0
    self._num_rigid_modes = 0

    self.k_reduced: np.ndarray
    self.m_reduced: np.ndarray
    self.eigenvalues : np.ndarray
    self.modes: np.ndarray
    self._stiffness_matrix: csr_matrix
    self._mass_matrix: csr_matrix


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
    k_buffer = dok_matrix((self._full_size, self._full_size), dtype=np.double)
    m_buffer = dok_matrix((self._full_size, self._full_size), dtype=np.double)

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

    print('Mass-spring model building complete\n')


  def _build_linear_fem_model(self):
    '''Internal method to build matrices for the linearelastic FEM model.'''
    print('Building Linearelastic FEM model...')
    # TODO
    print('Linearelastic FEM model building complete\n')


  def build_matrice(self, cons_model):
    '''Build full-space matrices stiffness and mass matrices for the given mesh.'''

    self._full_size = self._mesh.num_nodes * 3

    if cons_model == 0:
      self._build_mass_spring_model()

    elif cons_model == 1:
      self._build_linear_fem_model()

    else:
      raise RuntimeError('Invalid cons_model arguments.')

  def get_nodal_mass(self):
    '''Get the diagonal nodal mass matrix'''

    diagonal_mass = np.diag(self._mass_matrix.toarray())
    diagonal_mass = np.reshape(diagonal_mass, (self._mesh.num_nodes, 3))
    return diagonal_mass[:, 0]


  def eigen_solve(self, num_modes):
    '''Solve the generalized eigenvalue problems.'''

    # Since the first few (up to 6) are the rigid motion modes, we intentially request 6 mores
    # modes. At the end, we will count the number of rigid modes and disgard them.
    num_modes += 6

    if num_modes > self._full_size:
      # Clamp the #modes to be equal full size.
      num_modes = self._full_size
      print('[Warning] num_modes is larger than the maximum number of modes. ' +
            f'Clamp it to {self._full_size}.')
    elif num_modes < 0:
      # -1 means computing all the modes
      num_modes = self._full_size

    print('Solving eigenvalue problems...')
    if num_modes == self._full_size:
      # Run the dense eigenvalue problem solver if full modes are requested.
      print('[Warning] using the dense eigenvalue solver. Expect slower computation...')
      eigenvalues, eigenvectors = scipy.linalg.eigh(
                                    self._stiffness_matrix.todense(),
                                    self._mass_matrix.todense())
    else:
      # Run the sparse eigenvalue problem solver.
      # This should be more efficient than the dense solver.
      print('Sparse eigenvalue solver [eigsh] is used.')
      eigenvalues, eigenvectors = eigsh(
                                    self._stiffness_matrix,
                                    M=self._mass_matrix,
                                    k=num_modes,
                                    which='SM')
                                    # maxiter = 10000, tol=0)

    self.eigenvalues = eigenvalues
    self.modes = eigenvectors

    # get the number of rigid modes
    print('Solving complete!\nFirst 10 eigenvalues before the clean up are:')
    first_ten_eigenvalues = eigenvalues[:10]
    print(first_ten_eigenvalues)
    self._num_rigid_modes = first_ten_eigenvalues[first_ten_eigenvalues < 1e-6].size


  def compute_reduced_matrices(self, num_modes):
    '''Compute the reduced mass and stiffness matrices

    The reduced stiffness matrix and the reduced mass matrix is computed here. By virtue of the
    mass-orthogonality of the eigenvectors, the reduced matrices are diagonal. What's even better
    is that the reduced mass matrix is identity! There might be small numbers (near the round-off)
    at the off-diagonal entries, but they can be ignored.
    '''

    self.k_reduced = np.diag(self.modes.transpose() @ self._stiffness_matrix @ self.modes)
    self.m_reduced = np.diag(self.modes.transpose() @ self._mass_matrix @ self.modes)

    # remove the rigid modes eigenvalues and eigenvectors
    self.eigenvalues = self.eigenvalues[self._num_rigid_modes:self._num_rigid_modes+num_modes]
    self.modes = self.modes[:, self._num_rigid_modes:self._num_rigid_modes+num_modes]

    # only save non-rigid reduced mass and stiffness matrices
    self.k_reduced = self.k_reduced[self._num_rigid_modes:self._num_rigid_modes+num_modes]
    self.m_reduced = self.m_reduced[self._num_rigid_modes:self._num_rigid_modes+num_modes]


'''Modal analysis driver class.'''
class AnalysisDriver:

  def __init__(self, scale):
    self._mesh = Mesh()
    self._analyzer = ModalAnalyzer(self._mesh)
    self._scale_mesh = scale


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

    self._mesh.scale_mesh_to_solve(self._scale_mesh)


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
    self._analyzer.compute_reduced_matrices(num_modes)


  def _write_binary(self, filename, data, size):
    '''Write an array (double-precision) to the binary file.

    Args:
        filename: output file name, e.g., modes.bin.
        data: the array to be written to file.
        size: the number of entries to be written
    '''
    with open(filename, 'wb') as outfile:
      outfile.write(np.uint32(size))
      outfile.write(data.tobytes())


  def write_reduced_files(self):
    '''Write the reduced deformable files.

    This function writes the reduced stiffness matrix, nodal mass array,
    modes and eigenvalues to files. All output files are .bin files.
    '''
    # write eigenvalues
    self._write_binary('eigenvalues.bin',
                       np.ascontiguousarray(self._analyzer.eigenvalues, dtype=np.double),
                       self._analyzer.eigenvalues.size)

    # scale modes back to original dimensions
    self._analyzer.modes /= float(self._scale_mesh)

    # flatten the 2D array (ndofs x num_modes) of modes to a 1D array
    # we need to use column-major order because each column is a mode
    flat_mode = self._analyzer.modes.flatten('F')
    total_size = flat_mode.size
    # write modes
    self._write_binary('modes.bin',
                       np.ascontiguousarray(flat_mode, dtype=np.double),
                       total_size)

    # write reduced stiffness matrix
    self._write_binary('K_r_diag_mat.bin',
                       np.ascontiguousarray(self._analyzer.k_reduced, dtype=np.double),
                       self._analyzer.k_reduced.size)

    # write reduced mass matrix
    self._write_binary('M_r_diag_mat.bin',
                       np.ascontiguousarray(self._analyzer.m_reduced, dtype=np.double),
                       self._analyzer.m_reduced.size)

    # write nodal mass matrix
    nodal_mass = self._analyzer.get_nodal_mass()
    self._write_binary('M_diag_mat.bin',
                       np.ascontiguousarray(nodal_mass, dtype=np.double),
                       nodal_mass.size)

  def debug_reduced_mode_shape(self, mode, outfile='frame_1.obj'):
    '''Output the deformed mode shape for the given mode.

    Write the undeformed shape and the deformed shape of the given mode as an .obj file. Two files
    will be generated: frame_0.obj is the undeformed shape, frame_1.obj is the deformed mode shape.
    This allows users to view the file in the 3D modeling or VFX software as consecutive frames.
    '''

    # write undeformed mesh
    self._mesh.write_obj(self._mesh.nodes_original, 'frame_0.obj')

    # get the deformed shape
    deformed_node = self._mesh.nodes_original.copy()
    delta_nodes = np.reshape(self._analyzer.modes[:, mode-1], (self._mesh.num_nodes, 3))
    deformed_node += delta_nodes

    # write deformed mesh
    self._mesh.write_obj(deformed_node, outfile)

def main():
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mesh', action='store', type=str,
                        help='name of the input mesh', required=True)
    parser.add_argument('--num_modes', action='store', type=int, default=-1,
                        help='Number of modes to compute')
    parser.add_argument('--cons_model', action='store', type=int, default=0,
                        help='Constitutive model: 0 is mass-spring, 1 is linear FEM')
    parser.add_argument('--debug_mode', action='store', type=int, default=0,
                        help='check the mode shape of the given mode')
    parser.add_argument('--scale_mesh', action='store', type=float, default=1,
                        help='scale the mesh when computing the mode shape')
    args = parser.parse_args()

    driver = AnalysisDriver(args.scale_mesh)
    # load mesh from file
    driver.load_mesh(args.input_mesh)
    # run modal analysis
    driver.analyze(args.cons_model, args.num_modes)
    # write reduced files
    driver.write_reduced_files()

    # write the shape
    if args.debug_mode > 0:
      driver.debug_reduced_mode_shape(args.debug_mode)
    elif args.debug_mode == -1:
      for i in range(args.num_modes):
        driver.debug_reduced_mode_shape(i+1, f'frame_{i+1}.obj')

  except RuntimeError as err:
    print(err.args[0])


if __name__ == '__main__':
  main()
