@vite(['resources/css/app.css', 'resources/js/app.js'])

@props([

])


<div class='relative group w-xs lg:w-sm overflow-hidden font-secondaryAndButton'>
    <div class='bg-primary-85 group-hover:bg-secondary-relaxed-10 group-hover:border group-hover:border-secondary-relaxed-100 duration-100 w-18 h-18 p-3 relative z-10 flex flex-col items-center justify-center lg:w-20 lg:h-20 rounded-md '>
        <div class='w-10 rounded-full bg-secondary-relaxed-10 p-2 group-hover:bg-secondary-relaxed-60'>
            <img src="{{ asset('assets/icons/home.svg') }}" alt="home">
        </div>
        <p class='text-white text-micro lg:text-small group-hover:text-primary-70 group-hover:font-bold'>Home</p>
    </div>
    <div class='flex items-center justify-center w-18 h-18 lg:w-20 lg:h-20 bg-secondary-relaxed-20 rounded-full absolute z-5 top-1/2 left-0 -translate-y-1/2 group-hover:translate-x-1/2 group-hover:animate-spin duration-500 transition-transform'>
        <img src="{{ asset('assets/logo-no-text.png') }}" alt="" class='w-9 lg:w-10 p-1 bg-primary-70 rounded-full'>
    </div>
    <div class='bg-secondary-relaxed-60 w-max h-max rounded-md pl-29 lg:pl-33 px-3 py-2 absolute z-3 inset-0 top-1/2 left-0 -translate-x-full -translate-y-1/2 transition-transform duration-500 group-hover:translate-x-0'>
        <p>Give some social space</p>
    </div>
</div>

<div class='rounded-md mt-5 ml-5 w-40 py-2 px-4 overflow-hidden bg-amber-50 hover:bg-amber-200 flex flex-row items-center justify-start gap-3 font-secondaryAndButton group'>
    <div class='w-10 h-10 group-hover:bg-secondary-happy-10 group-hover:rounded-full p-1.5'>
        <img src="{{ asset('assets/icons/home.svg') }}" alt="home">
    </div>
    <div class='w-max h-max'>
        <p>Home</p>
    </div>
</div>