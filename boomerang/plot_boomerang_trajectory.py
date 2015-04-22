''' Plot animation of the Boomerang. '''

import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import sys
sys.path.append('..')
import vtk

import boomerang as bm
from config_local import DATA_DIR
from quaternion_integrator.quaternion import Quaternion
from utils import read_trajectory_from_txt


N_SPHERES = len(bm.M)
TIME_SKIP = 5
WRITE = True

class vtkTimerCallback():
  def __init__(self):
    self.timer_count = 0
    self.n = 1
 
  def execute(self,obj,event):
    print self.timer_count
    r_vectors = bm.get_boomerang_r_vectors_15(
      self.locations[self.n*TIME_SKIP], 
      Quaternion(self.orientations[self.n*TIME_SKIP]))

    for k in range(N_SPHERES):
      self.sources[k].SetCenter(r_vectors[k][0], r_vectors[k][1],
                                r_vectors[k][2])
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
  data_file_name = os.path.join(DATA_DIR, 'boomerang', data_name)
  
  params, locations, orientations = read_trajectory_from_txt(data_file_name)

  print 'Parameters are : ', params

  initial_r_vectors = bm.get_boomerang_r_vectors_15(
    locations[0], Quaternion(orientations[0]))

  # Create blobs
  blob_sources = []
  for k in range(N_SPHERES):
    blob_sources.append(vtk.vtkSphereSource())
    blob_sources[k].SetCenter(initial_r_vectors[0][0],
                              initial_r_vectors[0][1],
                              initial_r_vectors[0][2])
    blob_sources[k].SetRadius(bm.A)

  wall_source = vtk.vtkCubeSource()
  wall_source.SetCenter(0., 0., -0.125)
  wall_source.SetXLength(15.)
  wall_source.SetYLength(15.)
  wall_source.SetZLength(0.25)


  #Create a blob mappers and blob actors
  blob_mappers = []
  blob_actors = []
  for k in range(N_SPHERES):
    blob_mappers.append(vtk.vtkPolyDataMapper())
    blob_mappers[k].SetInputConnection(blob_sources[k].GetOutputPort())
    blob_actors.append(vtk.vtkActor())
    blob_actors[k].SetMapper(blob_mappers[k])
    blob_actors[k].GetProperty().SetColor(0, 0, 1)

  # Set up wall actor and mapper
  wall_mapper = vtk.vtkPolyDataMapper()
  wall_mapper.SetInputConnection(wall_source.GetOutputPort())
  wall_actor = vtk.vtkActor()
  wall_actor.SetMapper(wall_mapper)
  wall_actor.GetProperty().SetColor(0.5, 0.5, 0.5)

  # Create camera
  camera = vtk.vtkCamera()
  camera.SetPosition(0., -25., 5.)
  camera.SetFocalPoint(0., 0., 0.)
  camera.SetViewAngle(38.)

  # Setup a renderer, render window, and interactor
  renderer = vtk.vtkRenderer()
  renderer.SetActiveCamera(camera)
  renderWindow = vtk.vtkRenderWindow()
#  renderWindow.SetSize(1000, 1000)

  renderWindow.AddRenderer(renderer);
  renderWindowInteractor = vtk.vtkRenderWindowInteractor()
  renderWindowInteractor.SetRenderWindow(renderWindow)

  #Add the actors to the scene
  for k in range(len(bm.M)):
    renderer.AddActor(blob_actors[k])

  renderer.AddActor(wall_actor)
  renderer.SetBackground(0.9, 0.9, 0.9) # Background color off white

  #Render and interact
  renderWindow.Render()
  
  # Initialize must be called prior to creating timer events.
  renderWindowInteractor.Initialize()

  # Set up writer for pngs so we can save a movie.
  w2if = vtk.vtkWindowToImageFilter()
  w2if.SetInput(renderWindow)
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
  cb.locations = locations
  cb.orientations = orientations
  renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
  timerId = renderWindowInteractor.CreateRepeatingTimer(300);
  
  #start the interaction and timer
  renderWindowInteractor.Start()

