extern crate opencl3;


use opencl3::device::CL_DEVICE_TYPE_GPU;
use opencl3::platform::get_platforms;
use opencl3::device::Device;
use opencl3::context::Context;
use opencl3::program::Program;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::types::cl_float;

//------------------------Implmeting kernel in the same file----------------------------------------
const PROGRAM_SOURCE : &str = r#"
kernel void saxpy_float (global float* z, 
                        global float const* x,
                        global float const* y, 
                        float a)

	size_t i = get_global_id(0);
	z[i] = a * x[i] + y[i];
"#;
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

	// load du program source
	let source_program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "");

	//Création de la queue relié au context du device
	let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE).expect("Queue creation failed");

	/*----------------------------------Initialisation des données pour le calcul------------------------------------------*/
	const ARRAY_SIZE : usize = 10000;

	let ones : [cl_float; ARRAY_SIZE] = [1.0; ARRAY_SIZE];
	let mut sums: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
    for i in 0..ARRAY_SIZE {
        sums[i] = 1.0 + 1.0 * i as cl_float;
    }
	




}