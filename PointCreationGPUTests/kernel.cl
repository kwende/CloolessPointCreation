
typedef struct _Point3D
{
    float X; 
    float Y; 
    float Z;
} Point3D;

kernel void ComputePoints(
    global read_only ushort* depthValues, 
    global read_only float* affines,
    global write_only Point3D* projectivePoints, 
    global write_only Point3D* realWorldPoints, 
    global write_only Point3D* bedPoints,
    global write_only Point3D* floorPoints,
    int width,
    float inverseRotatedFx,
    float rotatedCx,
    float inverseRotatedFy,
    float rotatedCy)
{
    int index = get_global_id(0);

    ushort depth = depthValues[index]; 
    
    //if(depth > 0)
    { 
        float y = index / width;
        float x = index - (y * width);
        float z = depth;

        projectivePoints[index].X = x; 
        projectivePoints[index].Y = y; 
        projectivePoints[index].Z = z; 

        x = z * (x - rotatedCx) * inverseRotatedFx; 
        y = z * (rotatedCy - y) * inverseRotatedFy; 
        z = z; 

        realWorldPoints[index].X = x; 
        realWorldPoints[index].Y = y; 
        realWorldPoints[index].Z = z; 

        float realX = x; 
        float realY = y; 
        float realZ = z; 

        x = (affines[0] * realX + affines[1] * realY + affines[2] * realZ) + affines[3]; 
        y = (affines[4] * realX + affines[5] * realY + affines[6] * realZ) + affines[7]; 
        z = (affines[8] * realX + affines[9] * realY + affines[10] * realZ) + affines[11]; 

        bedPoints[index].X = x; 
        bedPoints[index].Y = y; 
        bedPoints[index].Z = z; 

        x = (affines[12] * realX + affines[13] * realY + affines[14] * realZ) + affines[15]; 
        y = (affines[16] * realX + affines[17] * realY + affines[18] * realZ) + affines[19]; 
        z = (affines[20] * realX + affines[21] * realY + affines[22] * realZ) + affines[23]; 

        floorPoints[index].X = x; 
        floorPoints[index].Y = y; 
        floorPoints[index].Z = z; 
    }
}
