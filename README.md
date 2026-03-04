# Software-based 3D Graphics Renderer #

Project developed as an engineering thesis and later continued as part of my master's thesis. This renderer is based on a triangle rasterization method and offers features such as backface culling, a depth buffer, perspective-correct texture mapping, and Phong shading. The application provides the ability to configure certain renderer options, such as the number of utilized threads, lightning, as well as displaying statistics for individual stages of the rendering process. Particular emphasis was placed on the use of SIMD vector instructions, balanced workload distribution across multiple threads, and tile-based rasterization. The final results were evaluated in terms of performance and scalability, confirming that the combination of these techniques significantly improves rendering efficiency.

## <b>Application overview</b>

https://github.com/user-attachments/assets/4156d527-2ac0-40f3-a19b-441d89f1380b

## <b>Screenshots</b>

### Model A - 8 884 triangles
<img width="1920" height="1056" alt="obraz" src="https://github.com/user-attachments/assets/8332557c-a7af-4207-8473-bcb2042dccf3" />

### Model B - 52 847 triangles
<img width="1920" height="1058" alt="obraz" src="https://github.com/user-attachments/assets/1114d70f-d307-4796-9289-903fa2a88f06" />

### Wireframe
<img width="1917" height="1061" alt="obraz" src="https://github.com/user-attachments/assets/32d327a3-f4bf-49d4-85fd-5f1ce1e4325c" />

### Colorize threads option
<img width="1922" height="1060" alt="obraz" src="https://github.com/user-attachments/assets/db8ad0a2-594e-4934-9633-74a8054797e9" />

## <b>Performance charts</b>

<img width="1313" height="716" alt="obraz" src="https://github.com/user-attachments/assets/c1b33881-ed0a-42c6-aa9a-1e8410dfcb27" />
<img width="1310" height="693" alt="obraz" src="https://github.com/user-attachments/assets/36a8e43e-9883-4ad5-82bd-2f6ce4b6a7b6" />

• ```CPU``` version – a tiled version of the reference algorithm implemented as a part of the engineering thesis (without the additional optimizations introduced in the master’s thesis - SIMD, etc.),  
• ```CPUx8``` version – using a SIMD-based drawing function through vectorization emulation, processing 8 values per loop iteration,  
• ```SSEx4``` version – using SSE instructions to process 4 pixels simultaneously,  
• ```SSEx8``` version – using SSE instructions to process 8 pixels by issuing two SSE instruction calls per operation,  
• ```AVXx8``` version – using AVX instructions to process 8 pixels simultaneously.
