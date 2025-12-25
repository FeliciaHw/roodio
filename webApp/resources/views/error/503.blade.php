<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>503 Service Unavailable</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>
<body class='bg-primary-100 h-screen flex items-center justify-center mx-8'>
    <div class=' flex flex-col w-full h-max items-center md:flex-row md:justify-center'>
        <div class='w-72 mb-1 md:w-80'>
            <img src="{{ asset("assets/503.png") }}" alt="503">
        </div>
        <div class='font-secondaryAndButton text-small text-white text-center md:text-justify md:ml-8 md:text-body'>
            <p class='font-primary text-body font-bold text-warning-lighten mb-1 md:text-paragraph'>503 | Service Unavailable</p>
            <p>Website is under construction or maintenance. Sorry for the inconvenient. We will back soon!</p>
        </div>
    </div>
</body>
</html>