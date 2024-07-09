extern crate opencl3;


use opencl3::device::CL_DEVICE_TYPE_GPU;
use opencl3::platform::get_platforms;
use opencl3::device::Device;
use opencl3::context::Context;
use opencl3::program::Program;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::types::{cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;

//------------------------Implmeting kernel in the same file----------------------------------------
const PROGRAM_SOURCE : &str = r#"
kernel void saxpy_float (global float* z, 
                        global float const* x,
                        global float const* y, 
                        float a)
{
	size_t i = get_global_id(0);
	z[i] = a * x[i] + y[i];
}"#;
const KERNEL_NAME: &str = "saxpy_float";
//--------------------------------------------------------------------------------------------------


fn main (){
	let platforms = get_platforms().unwrap();
	assert!(0 < platforms.len());

	// Get first platform
	let platform = &platforms[0];

	// Get devices of the platform
	let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
	assert!(devices.len() > 0 );

	// Getting the avaible device

	let device = Device::new(devices[0]);
	//println!("GPU name is : {:?}", device.board_name_amd());

	/*----------------------------------Initialisation du environement openCL------------------------------------------*/
	
	//création du contexte
	let context = Context::from_device(&device).expect("Context frome device failed");

	let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");

    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

	//Création de la queue relié au context du device
	let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE).expect("Queue creation failed");

	/*----------------------------------Initialisation des données pour le calcul------------------------------------------*/
	const ARRAY_SIZE : usize = 1000;

	let ones : [cl_float; ARRAY_SIZE] = [1.0; ARRAY_SIZE];
	let mut sums: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
    for i in 0..ARRAY_SIZE {
        sums[i] = 1.0 + 1.0 * i as cl_float;
    }

	// creating openCL buffers

	// X & Y input buffers
	let mut x = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut()).unwrap()};
    let mut y = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut()).unwrap()};

	// Z output buffer
    let z = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut()).unwrap()};

	// Blocking write
	let _x_write_event = unsafe { queue.enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &ones, &[]).unwrap() };

	// Non-blocking write, wait for y_write_event
    let y_write_event =
        unsafe { queue.enqueue_write_buffer(&mut y, CL_NON_BLOCKING, 0, &sums, &[]).unwrap() };

	// a value for the kernel function
	let a: cl_float = 300.0;

	// Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&z)
            .set_arg(&x)
            .set_arg(&y)
            .set_arg(&a)
            .set_global_work_size(ARRAY_SIZE)
            .set_wait_event(&y_write_event)
            .enqueue_nd_range(&queue).unwrap()
    };

	let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

	// Récupérer les résultats
	let mut results: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
	let _event = unsafe { queue.enqueue_read_buffer(&z, CL_NON_BLOCKING, 0, &mut results, &events).unwrap()};

	// Block until all commands on the queue have completed
    queue.finish().unwrap();

	assert_eq!(1300.0, results[ARRAY_SIZE - 1]);
    println!("results back: {}", results[ARRAY_SIZE - 1]);

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start().unwrap();
    let end_time = kernel_event.profiling_command_end().unwrap();
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}", duration);

}