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
      if len(new_input) != 4:
        print('The input needs five values variables.')
        new_input = ask_new_input()
      else:
        invalid_input = False
        th[0] = new_input[0]
        th[2] = -new_input[1]
        th[4] = new_input[2]
        th[5] = -new_input[3] + 90
        th[6] = new_input[3] + 90
# ******************************************************************************

# Introducción de los valores de las articulaciones
if len(sys.argv) == 1:
  sys.exit('The expected parameters are: θ1 θ2 θ3 θ4')
sys.argv.pop(0)
parameters = sys.argv
values = [float(value) for value in parameters]

# Parámetros D-H:
#              0' 1           1'   2          3                41              42      EF
d  = [         5, 2,          0,  -2,         5,                0,              0,      0]
th = [ values[0], 0, -values[1], -90, -values[2], -values[3] + 90, values[3] + 90,    -90]
a  = [         0, 0,          3,   0,         0,                1,              1,     -1]
al = [       -90, 0,          0, -90,        90,                0,              0,      0]

while True:
  # Orígenes para cada articulación
  origin = [0, 0, 0, 1]

  # Cálculo matrices transformación
  T00p = matriz_T(d[0], th[0], a[0], al[0])

  T0p1 = matriz_T(d[1], th[1], a[1], al[1])
  T01 = np.dot(T00p, T0p1)

  T11p = matriz_T(d[2], th[2], a[2], al[2])
  T01p = np.dot(T01, T11p)

  T1p2 = matriz_T(d[3], th[3], a[3], al[3])
  T02 = np.dot(T01p, T1p2)

  T23 = matriz_T(d[4], th[4], a[4], al[4])
  T03 = np.dot(T02, T23)

  T341 = matriz_T(d[5], th[5], a[5], al[5])
  T041 = np.dot(T03, T341)

  T342 = matriz_T(d[6], th[6], a[6], al[6])
  T042 = np.dot(T03, T342)

  T3EF = matriz_T(d[7], th[7], a[7], al[7])
  T0EF = np.dot(T03, T3EF)

  # Transformación de cada articulación
  o00 = origin
  o0p0 = np.dot(T00p, origin).tolist()
  o10 = np.dot(T01, origin).tolist()
  o1p0 = np.dot(T01p, origin).tolist()
  o20 = np.dot(T02, origin).tolist()
  o30 = np.dot(T03, origin).tolist()
  o410 = np.dot(T041, origin).tolist()
  o420 = np.dot(T042, origin).tolist()
  oEF = np.dot(T0EF, origin).tolist()

  # Mostrar resultado de la cinemática directa
  muestra_origenes([o00, o0p0, o10, o1p0, o20, o30, [[o410], [o420]]], oEF)
  muestra_robot([o00, o0p0, o10, o1p0, o20, o30, [[o410], [o420]]], oEF)

  new_input = ask_new_input()
  update_manipulator(new_input)