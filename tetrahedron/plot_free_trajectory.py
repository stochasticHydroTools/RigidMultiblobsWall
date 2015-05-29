''' Plot animation of the free tetrahedron trajectory. '''

import numpy as np
import os
import sys
sys.path.append('..')
import vtk


from config_local import DATA_DIR
from quaternion_integrator.quaternion import Quaternion
import tetrahedron_free as tf
from utils import read_trajectory_from_txt_old

N_SPHERES = 4
TIME_SKIP = 5
WRITE = True

class vtkTimerCallback():
  def __init__(self):
    self.timer_count = 0
    self.n = 1
 
  def execute(self,obj,event):
    print self.timer_count
    r_vectors = tf.get_free_r_vectors(
      self.locations[self.n*TIME_SKIP], 
      Quaternion(self.orientations[self.n*TIME_SKIP]))

    for k in range(N_SPHERES):
      self.sources[k].SetCenter(r_vectors[k][0], r_vectors[k][1],
                                r_vectors[k][2])
    for k in range(N_SPHERES):
      for j in range(0, k):
        self.line_sources[k][j].SetPoint1(r_vectors[k][0], 
                                          r_vectors[k][1], 
                                          r_vectors[k][2])
        self.line_sources[k][j].SetPoint2(r_vectors[j][0], 
                                          r_vectors[j][1], 
                                          r_vectors[j][2])

    iren = obj
    iren.GetRenderWindow().Render()
    if WRITE:
      self.w2if.Update()
      self.w2if.Modified()
      self.lwr.SetFileName(os.path.join(
          '.', 'figures',
          'frame'+ ('%03d' % self.n)+'.png'))
      self.lwr.Write()
    self.timer_count += 0.1
    self.n += 1


if __name__ == '__main__':
  # Data file name where trajectory data is stored.
  data_name = sys.argv[1]

  #######
  data_file_name = os.path.join(DATA_DIR, 'tetrahedron', data_name)
  
  params, locations, orientations = read_trajectory_from_txt_old(data_file_name)

  print 'Parameters are : ', params

  initial_r_vectors = tf.get_free_r_vectors(
    locations[0], Quaternion(orientations[0]))

  # Create blobs
  blob_sources = []
  for k in range(N_SPHERES):
    blob_sources.append(vtk.vtkSphereSource())
    blob_sources[k].SetCenter(initial_r_vectors[0][0],
                              initial_r_vectors[0][1],
                              initial_r_vectors[0][2])
    blob_sources[k].SetRadius(tf.A)

  wall_source = vtk.vtkCubeSource()
  wall_source.SetCenter(0., 0., -0.125)
  wall_source.SetXLength(10.)
  wall_source.SetYLength(10.)
  wall_source.SetZLength(0.25)


  #Create a blob mappers and blob actors
  blob_mappers = []
  blob_actors = []
  for k in range(N_SPHERES):
    blob_mappers.append(vtk.vtkPolyDataMapper())
    blob_mappers[k].SetInputConnection(blob_sources[k].GetOutputPort())
    blob_actors.append(vtk.vtkActor())
    blob_actors[k].SetMapper(blob_mappers[k])
    blob_actors[k].GetProperty().SetColor(1, 0, 0)

  # Set up wall actor and mapper
  wall_mapper = vtk.vtkPolyDataMapper()
  wall_mapper.SetInputConnection(wall_source.GetOutputPort())
  wall_actor = vtk.vtkActor()
  wall_actor.SetMapper(wall_mapper)
  wall_actor.GetProperty().SetColor(0.3, 0.95, 0.3)

  # Create lines
  line_sources = []
  for k in range(N_SPHERES):
    line_sources.append([])
    for j in range(0, k):
      line_sources[k].append(vtk.vtkLineSource())
      line_sources[k][j-1].SetPoint1(initial_r_vectors[k][0], 
                                     initial_r_vectors[k][1], 
                                     initial_r_vectors[k][2])
      line_sources[k][j-1].SetPoint2(initial_r_vectors[j][0], 
                                     initial_r_vectors[j][1], 
                                     initial_r_vectors[j][2])

  line_mappers = []
  line_actors = []
  for k in range(N_SPHERES):
    line_mappers.append([])
    line_actors.append([])
    for j in range(0, k):
      line_mappers[k].append(vtk.vtkPolyDataMapper())
      line_mappers[k][j].SetInputConnection(
        line_sources[k][j].GetOutputPort())
      line_actors[k].append(vtk.vtkActor())
      line_actors[k][j].SetMapper(line_mappers[-1][-1])
      line_actors[k][j].GetProperty().SetColor(1, 0, 0)

  # Create camera
  camera = vtk.vtkCamera()
  camera.SetPosition(0., -15., 3.)
  camera.SetFocalPoint(0., 0., 0.)
  camera.SetViewAngle(37.)

  # Setup a renderer, render window, and interactor
  renderer = vtk.vtkRenderer()
  renderer.SetActiveCamera(camera)
  renderWindow = vtk.vtkRenderWindow()
  renderWindow.SetSize(1000, 1000)

  renderWindow.AddRenderer(renderer);
  renderWindowInteractor = vtk.vtkRenderWindowInteractor()
  renderWindowInteractor.SetRenderWindow(renderWindow)

  #Add the actors to the scene
  for k in range(N_SPHERES):
    renderer.AddActor(blob_actors[k])
  
  # Add Line Ators to the scene.
  for k in range(N_SPHERES):
    for j in range(0, k):
      renderer.AddActor(line_actors[k][j])

  renderer.AddActor(wall_actor)

  renderer.SetBackground(0.9, 0.9, 0.9) # Background color off white

  #Render and interact
  renderWindow.Render()
  
  # Initialize must be called prior to creating timer events.
  renderWindowInteractor.Initialize()

  # Set up writer for pngs so we can save a movie.
  w2if = vtk.vtkWindowToImageFilter()
  w2if.SetInput(renderWindow)
  w2if.SetMagnification(1.5)
  w2if.Update()
  w2if.ReadFrontBufferOff()
  lwr = vtk.vtkPNGWriter()
  lwr.SetInput( w2if.GetOutput() )

  # Sign up to receive TimerEvent
  cb = vtkTimerCallback()
  cb.actors = blob_actors
  cb.lwr = lwr
  cb.w2if = w2if
  cb.sources = blob_sources
  cb.line_sources = line_sources
  cb.locations = locations
  cb.orientations = orientations
  renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
  timerId = renderWindowInteractor.CreateRepeatingTimer(300);
  
  #start the interaction and timer
  renderWindowInteractor.Start()

# if __name__ == '__main__':
#   # Data file name where trajectory data is stored.
#   data_name = sys.argv[1]


#   #######
#   data_file_name = os.path.join(DATA_DIR, 'tetrahedron', data_name)
  
#   params, locations, orientations = read_trajectory_from_txt(data_file_name)

#   fig = plt.figure()
#   ax = Axes3D(fig) #fig.add_axes([0.1, 0.1, 0.8, 0.8])

#   ax.set_xlim3d([-10., 10.])
#   ax.set_ylim3d([-10., 10.])
#   ax.set_zlim3d([-0.5, 8.])
  
#   wall_x = [-10. + k*20./20. for k in range(20) ]
#   for k in range(19):
#     wall_x += [-10. + k*20./20. for k in range(20) ]

#   wall_y = [-10. for _ in range(20)]
#   for k in range(19):
#     wall_y += [-10 + k*20./20. for _ in range(20)]


#   wall, = ax.plot(wall_x, wall_y, np.zeros(400), 'k.')
#   blobs, = ax.plot([], [], [], 'bo', ms=24)
#   connectors = [0]*12
#   for j in range(4):
#     for k in range(j+1, 4):
#       connector, = ax.plot([], [], [], 'b-', lw=2)
#       connectors[j*3 + k] = connector


#   def init_animation():
#     ''' Initialize 3D animation. '''
#     r_vectors = tf.get_free_r_vectors([0., 0., tf.H], Quaternion([1., 0., 0., 0.]))
#     blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
#                    [r_vectors[k][1] for k in range(len(r_vectors))])
#     blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
#     for j in range(len(r_vectors)):
#       for k in range(j+1, len(r_vectors)):
#         connectors[j*3 + k].set_data([r_vectors[j][0],r_vectors[k][0]], 
#                                      [r_vectors[j][1],r_vectors[k][1]])
#         connectors[j*3 + k].set_3d_properties([r_vectors[j][2], r_vectors[k][2]])

    
#   def update(n):
#     ''' Update the tetrahedron animation '''
#     location = locations[n]
#     orientation = orientations[n]
#     r_vectors = tf.get_free_r_vectors(location, Quaternion(orientation))
#     blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
#                    [r_vectors[k][1] for k in range(len(r_vectors))])
#     blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
#     for j in range(len(r_vectors)):
#       for k in range(j+1, len(r_vectors)):
#         connectors[j*3 + k].set_data([r_vectors[j][0],r_vectors[k][0]], 
#                                      [r_vectors[j][1],r_vectors[k][1]])
#         connectors[j*3 + k].set_3d_properties([r_vectors[j][2], r_vectors[k][2]])
    
  
# anim = animation.FuncAnimation(fig, update, init_func=init_animation, 
#                                frames=700, interval=20, blit=True)
# anim.save('tetrahedron.mp4', writer='ffmpeg')
