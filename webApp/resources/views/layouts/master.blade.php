<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>@yield('title', 'ROODIO')</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
    @stack('style')
    @stack('script')
</head>
<body class="container-body @yield('bodyClass', '')">
    @yield('bodyContent')
</body>
</html>