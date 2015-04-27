''' Plot frame of Sphere162. '''

import numpy as np
import os
import sys
sys.path.append('..')
import vtk

import sphere162 as sph162
from config_local import DATA_DIR
from quaternion_integrator.quaternion import Quaternion
from utils import read_trajectory_from_txt


N_SPHERES = 162
TIME_SKIP = 10
WRITE = True
DRAW_COH = True


class vtkTimerCallback():
  def __init__(self):
    self.timer_count = 0
    self.n = 1
 
  def execute(self,obj,event):
    print self.timer_count
    r_vectors = sph162.get_sphere_162_blobs_r_vectors(
      self.locations[0], 
      self.orientations[0])

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

  #######
  orientations = [Quaternion([1., 0., 0., 0.])]
  locations = [[0., 0., 0.]]
  initial_r_vectors = sph162.get_sphere_162_blobs_r_vectors(locations[0], 
                                                            orientations[0])

  # Create blobs
  blob_sources = []
  for k in range(N_SPHERES):
    blob_sources.append(vtk.vtkSphereSource())
    blob_sources[k].SetCenter(initial_r_vectors[0][0],
                              initial_r_vectors[0][1],
                              initial_r_vectors[0][2])
    blob_sources[k].SetRadius(sph162.VERTEX_A)



  #Create a blob mappers and blob actors
  blob_mappers = []
  blob_actors = []
  for k in range(N_SPHERES):
    blob_mappers.append(vtk.vtkPolyDataMapper())
    blob_mappers[k].SetInputConnection(blob_sources[k].GetOutputPort())
    blob_actors.append(vtk.vtkActor())
    blob_actors[k].SetMapper(blob_mappers[k])
    blob_actors[k].GetProperty().SetColor(1, 0, 0)
  
  # Create camera
  camera = vtk.vtkCamera()
  # Close
  camera.SetPosition(0., -5., 4.)
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
  cb.orientations = orientations
  cb.locations = locations
  renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
  timerId = renderWindowInteractor.CreateRepeatingTimer(300);
  
  #start the interaction and timer
  renderWindowInteractor.Start()

