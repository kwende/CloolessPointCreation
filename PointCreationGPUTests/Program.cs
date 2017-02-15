using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Single;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Diagnostics;
using Cloo;

namespace PointCreationGPUTests
{
    struct Point3D
    {
        public Point3D(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }


    class Program
    {
        static void LoadVariables(
            out ushort[] depthPixels,
            out int width,
            out float inverseRotatedFx,
            out float rotatedCx,
            out float inverseRotatedFy,
            out float rotatedCy,
            out Matrix bedTransformationM,
            out Matrix bedTransformationb,
            out Matrix floorTransformationM,
            out Matrix floorTransformationb)
        {
            BinaryFormatter formatter = new BinaryFormatter();

            using (FileStream fin = File.OpenRead("depthpixels.dat"))
            {
                depthPixels = (ushort[])formatter.Deserialize(fin);
            }

            using (FileStream fin = File.OpenRead("blittables.dat"))
            {
                byte[] buffer = new byte[sizeof(float)];

                fin.Read(buffer, 0, sizeof(int));
                width = BitConverter.ToInt32(buffer, 0);

                fin.Read(buffer, 0, sizeof(float));
                inverseRotatedFx = BitConverter.ToSingle(buffer, 0);

                fin.Read(buffer, 0, sizeof(float));
                rotatedCx = BitConverter.ToSingle(buffer, 0);

                fin.Read(buffer, 0, sizeof(float));
                inverseRotatedFy = BitConverter.ToSingle(buffer, 0);

                fin.Read(buffer, 0, sizeof(float));
                rotatedCy = BitConverter.ToSingle(buffer, 0);
            }

            using (FileStream fin = File.OpenRead("bedTransformationM.dat"))
            {
                bedTransformationM = (Matrix)formatter.Deserialize(fin);
            }

            using (FileStream fin = File.OpenRead("bedTransformationM.dat"))
            {
                bedTransformationb = (Matrix)formatter.Deserialize(fin);
            }

            using (FileStream fin = File.OpenRead("bedTransformationM.dat"))
            {
                floorTransformationM = (Matrix)formatter.Deserialize(fin);
            }

            using (FileStream fin = File.OpenRead("bedTransformationM.dat"))
            {
                floorTransformationb = (Matrix)formatter.Deserialize(fin);
            }
        }

        public static Point3D CreateProjectivePoint(ushort depth, int index, int width)
        {
            float y = index / width;
            float x = index - (y * width);
            float z = depth;

            return new Point3D(
                x,
                y,
                z
            );
        }

        private static Point3D ConvertProjectiveToRealWorld(Point3D projPoint, float inverseRotatedFx, float inverseRotatedFy, float rotatedCx, float rotatedCy)
        {
            return new Point3D(
                projPoint.Z * (projPoint.X - rotatedCx) * inverseRotatedFx,
                projPoint.Z * (rotatedCy - projPoint.Y) * inverseRotatedFy,
                projPoint.Z);
        }

        static float ComputAverageSlowTime(
            ushort[] depthPixels,
            int width,
            float inverseRotatedFx,
            float rotatedCx,
            float inverseRotatedFy,
            float rotatedCy,
            Matrix bedTransformationM,
            Matrix bedTransformationb,
            Matrix floorTransformationM,
            Matrix floorTransformationb,
            int numberOfIterations)
        {
            Point3D[] projectivePoints = new Point3D[depthPixels.Length];
            Point3D[] realWorldPoints = new Point3D[depthPixels.Length];
            Point3D[] bedPoints = new Point3D[depthPixels.Length];
            Point3D[] floorPoints = new Point3D[depthPixels.Length];

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int c = 0; c < numberOfIterations; c++)
            {
                for (int i = 0; i < depthPixels.Length; i++)
                {
                    if (depthPixels[i] != 0)
                    {
                        // create projective points
                        projectivePoints[i] = CreateProjectivePoint(depthPixels[i], i, width);

                        // create real world points
                        realWorldPoints[i] = ConvertProjectiveToRealWorld(projectivePoints[i], inverseRotatedFx, inverseRotatedFy, rotatedCx, rotatedCy);

                        Point3D point = realWorldPoints[i];

                        // create bed points
                        bedPoints[i] = new Point3D(
                                (bedTransformationM[0, 0] * point.X + bedTransformationM[0, 1] * point.Y + bedTransformationM[0, 2] * point.Z) + bedTransformationb[0, 0],
                                (bedTransformationM[1, 0] * point.X + bedTransformationM[1, 1] * point.Y + bedTransformationM[1, 2] * point.Z) + bedTransformationb[1, 0],
                                (bedTransformationM[2, 0] * point.X + bedTransformationM[2, 1] * point.Y + bedTransformationM[2, 2] * point.Z) + bedTransformationb[2, 0]);

                        // create floor points
                        floorPoints[i] = new Point3D(
                                (floorTransformationM[0, 0] * point.X + floorTransformationM[0, 1] * point.Y + floorTransformationM[0, 2] * point.Z) + floorTransformationb[0, 0],
                                (floorTransformationM[1, 0] * point.X + floorTransformationM[1, 1] * point.Y + floorTransformationM[1, 2] * point.Z) + floorTransformationb[1, 0],
                                (floorTransformationM[2, 0] * point.X + floorTransformationM[2, 1] * point.Y + floorTransformationM[2, 2] * point.Z) + floorTransformationb[2, 0]);
                    }
                }
            }

            sw.Stop();

            return sw.ElapsedMilliseconds / (numberOfIterations * 1.0f);
        }

        static float ComputeAverageGPUTime(
            ushort[] depthPixels,
            int width,
            float inverseRotatedFx,
            float rotatedCx,
            float inverseRotatedFy,
            float rotatedCy,
            Matrix bedTransformationM,
            Matrix bedTransformationb,
            Matrix floorTransformationM,
            Matrix floorTransformationb,
            int numberOfIterations)
        {
            // pick the device platform 
            ComputePlatform intelGPU = ComputePlatform.Platforms.Where(n => n.Name.Contains("Intel")).First();

            ComputeContext context = new ComputeContext(
                ComputeDeviceTypes.Gpu, // use the gpu
                new ComputeContextPropertyList(intelGPU), // use the intel openCL platform
                null,
                IntPtr.Zero);

            // the command queue is the, well, queue of commands sent to the "device" (GPU)
            ComputeCommandQueue commandQueue = new ComputeCommandQueue(
                context, // the compute context
                context.Devices[0], // first device matching the context specifications
                ComputeCommandQueueFlags.None); // no special flags

            string kernelSource = null;
            using (StreamReader sr = new StreamReader("kernel.cl"))
            {
                kernelSource = sr.ReadToEnd();
            }

            // create the "program"
            ComputeProgram program = new ComputeProgram(context, new string[] { kernelSource });

            // compile. 
            program.Build(null, null, null, IntPtr.Zero);
            ComputeKernel kernel = program.CreateKernel("ComputePoints");

            Point3D[] outProjectivePoints = new Point3D[depthPixels.Length];
            Point3D[] outRealPoints = new Point3D[depthPixels.Length];
            Point3D[] outBedPoints = new Point3D[depthPixels.Length];
            Point3D[] outFloorPoints = new Point3D[depthPixels.Length];

            float[] affines = new float[24];

            // do bed affines first because that's what assembly code expects
            int z = 0;
            for (int b = 0; b < 3; b++)
            {
                for (int c = 0; c < 3; c++)
                {
                    affines[z++] = bedTransformationM[b, c];
                }
                affines[z++] = bedTransformationb[b, 0];
            }

            // do floor affines next because that's what assembly code expects
            for (int b = 0; b < 3; b++)
            {
                for (int c = 0; c < 3; c++)
                {
                    affines[z++] = floorTransformationM[b, c];
                }
                affines[z++] = floorTransformationb[b, 0];
            }

            ComputeBuffer<float> affinesBuffer = new ComputeBuffer<float>(context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer,
                affines);
            kernel.SetMemoryArgument(1, affinesBuffer);

            ComputeBuffer<Point3D> projectivePointsBuffer = new ComputeBuffer<Point3D>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer,
                outProjectivePoints);
            kernel.SetMemoryArgument(2, projectivePointsBuffer);

            ComputeBuffer<Point3D> realPointsBuffer = new ComputeBuffer<Point3D>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer,
                outBedPoints);
            kernel.SetMemoryArgument(3, realPointsBuffer);

            ComputeBuffer<Point3D> bedPointsBuffer = new ComputeBuffer<Point3D>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer,
                outFloorPoints);
            kernel.SetMemoryArgument(4, projectivePointsBuffer);

            ComputeBuffer<Point3D> floorPointsBuffer = new ComputeBuffer<Point3D>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer,
                outRealPoints);
            kernel.SetMemoryArgument(5, realPointsBuffer);

            kernel.SetValueArgument<int>(6, width);
            kernel.SetValueArgument<float>(7, inverseRotatedFx);
            kernel.SetValueArgument<float>(8, rotatedCx);
            kernel.SetValueArgument<float>(9, inverseRotatedFy);
            kernel.SetValueArgument<float>(10, rotatedCy);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int c = 0; c < numberOfIterations; c++)
            {
                ComputeBuffer<ushort> depthPointsBuffer = new ComputeBuffer<ushort>(context,
                    ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer,
                    depthPixels);
                kernel.SetMemoryArgument(0, depthPointsBuffer);

                commandQueue.Execute(kernel, new long[] { 0 }, new long[] { depthPixels.Length }, null, null);

                unsafe
                {
                    fixed (Point3D* projectivePointsPtr = outProjectivePoints)
                    {
                        fixed(Point3D* realPointsPtr = outRealPoints)
                        {
                            fixed(Point3D* bedPointsPtr = outBedPoints)
                            {
                                fixed(Point3D* floorPointsPtr = outFloorPoints)
                                {
                                    commandQueue.Read(projectivePointsBuffer, false, 0, outProjectivePoints.Length, new IntPtr(projectivePointsPtr), null);
                                    commandQueue.Read(realPointsBuffer, false, 0, outProjectivePoints.Length, new IntPtr(realPointsPtr), null);
                                    commandQueue.Read(bedPointsBuffer, false, 0, outProjectivePoints.Length, new IntPtr(bedPointsPtr), null);
                                    commandQueue.Read(floorPointsBuffer, false, 0, outProjectivePoints.Length, new IntPtr(floorPointsPtr), null);
                                    commandQueue.Finish();
                                }
                            }
                        }
                    }
                }
            }
            sw.Stop();

            return sw.ElapsedMilliseconds / (numberOfIterations * 1.0f);
        }

        static void Main(string[] args)
        {
            ushort[] depthPixels;
            int width;
            float inverseRotatedFx;
            float rotatedCx;
            float inverseRotatedFy;
            float rotatedCy;
            Matrix bedTransformationM;
            Matrix bedTransformationb;
            Matrix floorTransformationM;
            Matrix floorTransformationb;

            LoadVariables(
                out depthPixels,
                out width,
                out inverseRotatedFx,
                out rotatedCx,
                out inverseRotatedFy,
                out rotatedCy,
                out bedTransformationM,
                out bedTransformationb,
                out floorTransformationM,
                out floorTransformationb);

            const int NumberOfIterations = 100;

            float averageGPUTime = ComputeAverageGPUTime(
                depthPixels,
                width,
                inverseRotatedFx,
                rotatedCx,
                inverseRotatedFy,
                rotatedCy,
                bedTransformationM,
                bedTransformationb,
                floorTransformationM,
                floorTransformationb,
                NumberOfIterations);

            float averageSlowTime = ComputAverageSlowTime(
                depthPixels,
                width,
                inverseRotatedFx,
                rotatedCx,
                inverseRotatedFy,
                rotatedCy,
                bedTransformationM,
                bedTransformationb,
                floorTransformationM,
                floorTransformationb,
                NumberOfIterations);

            Console.WriteLine($"Slow points averages {averageSlowTime} ms");
            Console.WriteLine($"GPU points averages {averageGPUTime} ms");
            Console.WriteLine($"GPU is {averageSlowTime / averageGPUTime}x faster."); 
            Console.ReadLine();

            return;
        }
    }
}
