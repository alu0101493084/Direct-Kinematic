#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática directa mediante Denavit-Hartenberg.

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ******************************************************************************
# Declaración de funciones

def ramal(I,prev=[],base=0):
  # Convierte el robot a una secuencia de puntos para representar
  O = []
  if I:
    if isinstance(I[0][0],list):
      for j in range(len(I[0])):
        O.extend(ramal(I[0][j], prev, base or j < len(I[0])-1))
    else:
      O = [I[0]]
      O.extend(ramal(I[1:],I[0],base))
      if base:
        O.append(prev)
  return O

def muestra_robot(O,ef=[]):
  # Pinta en 3D
  OR = ramal(O)
  OT = np.array(OR).T
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  # Bounding box cúbico para simular el ratio de aspecto correcto
  max_range = np.array([OT[0].max()-OT[0].min()
                       ,OT[1].max()-OT[1].min()
                       ,OT[2].max()-OT[2].min()
                       ]).max()
  Xb = (0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten()
     + 0.5*(OT[0].max()+OT[0].min()))
  Yb = (0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten()
     + 0.5*(OT[1].max()+OT[1].min()))
  Zb = (0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten()
     + 0.5*(OT[2].max()+OT[2].min()))
  for xb, yb, zb in zip(Xb, Yb, Zb):
     ax.plot([xb], [yb], [zb], 'w')
  ax.plot3D(OT[0],OT[1],OT[2],marker='s')
  ax.plot3D([0],[0],[0],marker='o',color='k',ms=10)
  if not ef:
    ef = OR[-1]
  ax.plot3D([ef[0]],[ef[1]],[ef[2]],marker='s',color='r')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()
  return

def arbol_origenes(O,base=0,sufijo=''):
  # Da formato a los origenes de coordenadas para mostrarlos por pantalla
  if isinstance(O[0],list):
    for i in range(len(O)):
      if isinstance(O[i][0],list):
        for j in range(len(O[i])):
          arbol_origenes(O[i][j],i+base,sufijo+str(j+1))
      else:
        print('(O'+str(i+base)+sufijo+')0\t= '+str([round(j,3) for j in O[i]]))
  else:
    print('(O'+str(base)+sufijo+')0\t= '+str([round(j,3) for j in O]))

def muestra_origenes(O,final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Orígenes de coordenadas:')
  arbol_origenes(O)
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def matriz_T(d,theta,a,alpha):
  # Calcula la matriz T (ángulos de entrada en grados)
  th=theta*pi/180
  al=alpha*pi/180
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]

def ask_new_input():
  new_input = input('Enter new values: ')
  if new_input == '':
    sys.exit()
  new_input = [float(value) for value in (new_input.split())]
  return new_input

def update_manipulator(new_input):
  invalid_input = True
  while invalid_input:
    if filename == 'manipulators/m1.txt':
      if len(new_input) != 2:
        print('The input needs two values for the theta variables.')
        new_input = ask_new_input()
      else:
        invalid_input = False
        manipulator[1] = new_input
    elif filename == 'manipulators/m2.txt':
      if len(new_input) != 3:
        print('The input needs values for theta1, L2, and theta3.')
        new_input = ask_new_input()
      else:
        invalid_input = False
        manipulator[1][0] = new_input[0]
        manipulator[2][1] = new_input[1]
        manipulator[1][2] = new_input[2]
# ******************************************************************************

# Introducción de los valores de las articulaciones
if len(sys.argv) == 1:
  sys.exit('The expected parameter is the file with the information about the manipulator.\n' +
           'For example: \'manipulator.txt\' ')
filename = sys.argv[1]

# Parámetros D-H:
# manipulator = [d,theta,a,alpha]
manipulator = []
with open(filename, 'r') as file:
  values = []
  for j in range(4):
    line = file.readline().split()
    values = [float(value) for value in line]
    manipulator.append(values)
  number_of_joints = len(values)

while True:
  # Orígenes para cada articulación
  origins = [[0,0,0,1] for i in range(number_of_joints)]

  # Cálculo matrices transformación
  result = matriz_T(manipulator[0][0], manipulator[1][0], manipulator[2][0], manipulator[3][0])
  T_matrices = [result]
  for i in range(1,number_of_joints):
    result2 = matriz_T(manipulator[0][i], manipulator[1][i], manipulator[2][i], manipulator[3][i])
    result = np.dot(result, result2)
    T_matrices.append(result)

  # Transformación de cada articulación
  new_origins = [[0,0,0,1]]
  for i in range(len(origins)):
    new_origins.append(np.dot(T_matrices[i], origins[i]).tolist())

  # Mostrar resultado de la cinemática directa
  muestra_origenes(new_origins)
  muestra_robot(new_origins)

  new_input = ask_new_input()
  update_manipulator(new_input)