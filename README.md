# Software-based 3D Graphics Renderer #

This renderer is based on the rasterization method and offers features such as discarding invisible triangles (frustum culling and back-face culling), a depth buffer (Z-Buffer), perspective-correct texture mapping and basic shading using the Phong method. The application provides the ability to configure certain renderer options, such as the number of utilized threads, lightning, as well as displaying statistics for individual stages of the rendering process. 
Particular emphasis was placed on the use of SIMD vector instructions, even distribution of work across multiple threads, and tile-based rasterization. The obtained results were evaluated in terms of performance and scalability, confirming that the combination of these techniques significantly improves rendering efficiency.

## <b>Overwiev</b>

https://github.com/user-attachments/assets/4156d527-2ac0-40f3-a19b-441d89f1380b

## <b>Screenshots</b>

### <b>Model - 8 884 triangles</b>
<img width="1920" height="1056" alt="obraz" src="https://github.com/user-attachments/assets/8332557c-a7af-4207-8473-bcb2042dccf3" />

### <b>Model - 52 847 triangles</b>
<img width="1920" height="1058" alt="obraz" src="https://github.com/user-attachments/assets/1114d70f-d307-4796-9289-903fa2a88f06" />

### <b>Wireframe</b>
<img width="1917" height="1061" alt="obraz" src="https://github.com/user-attachments/assets/32d327a3-f4bf-49d4-85fd-5f1ce1e4325c" />

### <b>Colorize threads</b>
<img width="1922" height="1060" alt="obraz" src="https://github.com/user-attachments/assets/db8ad0a2-594e-4934-9633-74a8054797e9" />

## <b>Performance</b>

<img width="1313" height="716" alt="obraz" src="https://github.com/user-attachments/assets/c1b33881-ed0a-42c6-aa9a-1e8410dfcb27" />
<img width="1310" height="693" alt="obraz" src="https://github.com/user-attachments/assets/36a8e43e-9883-4ad5-82bd-2f6ce4b6a7b6" />

• <b>CPU</b>   version – a tiled version of the reference algorithm implemented as part of the engineering thesis, without the additional optimizations introduced in the master’s thesis (SIMD, etc.),  
• <b>CPUx8</b> version – using a SIMD-based drawing function through vectorization emulation, processing 8 values per loop iteration,  
• <b>SSEx4</b> version – using SSE instructions to process 4 pixels simultaneously,  
• <b>SSEx8</b> version – using SSE instructions to process 8 pixels by issuing two SSE instruction calls per operation,  
• <b>AVXx8</b> version – using AVX instructions to process 8 pixels simultaneously.
