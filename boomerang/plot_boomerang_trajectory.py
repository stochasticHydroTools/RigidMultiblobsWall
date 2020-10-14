''' Plot animation of the Boomerang. '''
import numpy as np
import os
import sys
sys.path.append('..')
import vtk

import boomerang as bm
from config_local import DATA_DIR
from quaternion_integrator.quaternion import Quaternion
from general_application_utils import read_trajectory_from_txt


N_SPHERES = len(bm.M)
TIME_SKIP = 8
WRITE = True
DRAW_COH = False

class vtkTimerCallback():
  def __init__(self):
    self.timer_count = 0
    self.n = 1
 
  def execute(self,obj,event):
    print self.timer_count
    self.textActor.SetInput('Time: %s' % self.timer_count)
    r_vectors = bm.get_boomerang_r_vectors_15(
      self.locations[self.n*TIME_SKIP], 
      Quaternion(self.orientations[self.n*TIME_SKIP]))

    for k in range(N_SPHERES):
      self.sources[k].SetCenter(r_vectors[k][0], r_vectors[k][1],
                                r_vectors[k][2])
    if DRAW_COH:
      coh = bm.calculate_boomerang_coh(
        self.locations[self.n*TIME_SKIP], 
        Quaternion(self.orientations[self.n*TIME_SKIP]))
      cod = bm.calculate_boomerang_cod(
        self.locations[self.n*TIME_SKIP], 
        Quaternion(self.orientations[self.n*TIME_SKIP]))
      self.coh_source.SetCenter(coh)
      self.cod_source.SetCenter(cod)

    iren = obj
    iren.GetRenderWindow().Render()
    if WRITE:
      self.w2if.Update()
      self.w2if.Modified()
      self.lwr.SetFileName(os.path.join(
          '.', 'figures',
          'frame'+ ('%03d' % self.n)+'.png'))
      self.lwr.Write()
    self.timer_count += 0.01*TIME_SKIP
    self.n += 1


if __name__ == '__main__':
  # Data file name where trajectory data is stored.
  data_name = sys.argv[1]

  #######
  data_file_name = os.path.join(DATA_DIR, 'boomerang', data_name)
  
  params, locations, orientations = read_trajectory_from_txt(data_file_name)

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

  if DRAW_COH:
    coh_source = vtk.vtkSphereSource()
    coh_point = bm.calculate_boomerang_coh(
      locations[0], Quaternion(orientations[0]))
    coh_source.SetCenter(coh_point)
    coh_source.SetRadius(0.1)
    cod_source = vtk.vtkSphereSource()
    cod_point = bm.calculate_boomerang_cod(
      locations[0], Quaternion(orientations[0]))
    cod_source.SetCenter(coh_point)
    cod_source.SetRadius(0.1)

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
  
  if DRAW_COH:
    coh_mapper = vtk.vtkPolyDataMapper()
    coh_mapper.SetInputConnection(coh_source.GetOutputPort())
    coh_actor = vtk.vtkActor()
    coh_actor.SetMapper(coh_mapper)
    coh_actor.GetProperty().SetColor(5., 5., 1.)
    cod_mapper = vtk.vtkPolyDataMapper()
    cod_mapper.SetInputConnection(cod_source.GetOutputPort())
    cod_actor = vtk.vtkActor()
    cod_actor.SetMapper(cod_mapper)
    cod_actor.GetProperty().SetColor(0., 0., 1.)


  # Set up wall actor and mapper
  wall_mapper = vtk.vtkPolyDataMapper()
  wall_mapper.SetInputConnection(wall_source.GetOutputPort())
  wall_actor = vtk.vtkActor()
  wall_actor.SetMapper(wall_mapper)
  wall_actor.GetProperty().SetColor(0.1, 0.95, 0.1)

  # Create camera
  camera = vtk.vtkCamera()
  camera.SetPosition(0., -20., 5.)
  camera.SetFocalPoint(0., 0., 1.)
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
  for k in range(len(bm.M)):
    renderer.AddActor(blob_actors[k])

  if DRAW_COH:
    renderer.AddActor(coh_actor)
    renderer.AddActor(cod_actor)

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

  # Set up text
  textActor = vtk.vtkTextActor()
  textActor.GetTextProperty().SetFontSize (24)
  textActor.SetDisplayPosition(400, 120)
  renderer.AddActor2D(textActor)
  textActor.GetTextProperty().SetColor( 0.0, 0.0, 0.0 )
 

  # Sign up to receive TimerEvent
  cb = vtkTimerCallback()
  cb.actors = blob_actors
  cb.lwr = lwr
  cb.w2if = w2if
  cb.sources = blob_sources
  if DRAW_COH:
    cb.coh_source = coh_source
    cb.cod_source = cod_source
  cb.locations = locations
  cb.orientations = orientations
  cb.textActor = textActor
  renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
  timerId = renderWindowInteractor.CreateRepeatingTimer(300);
  
  #start the interaction and timer
  renderWindowInteractor.Start()
