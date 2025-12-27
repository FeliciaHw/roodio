<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Thread</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>
<body>
    <form action="{{ route('thread.store') }}" method="POST">
        @csrf
        <div class="mb-3">
            <label for="title">Title</label>
            <input type="text" name="title" class="outline-1" value="{{ old('title') }}">
            @error('title')
            {{ $message }}
            @enderror
        </div>
        <div class="mb-3">
            <label for="content">Content</label>
            <textarea class="border" name="content">{{ old('content') }}</textarea>
            @error('content')
            {{ $message }}
            @enderror
        </div>
        <div class="mb-3">
            <label for="isReplyable">isReplyable</label>
            <input type="checkbox" name="isReplyable" class="outline-1" value="1">
        </div>
        <button type="submit">Submit</button>
    </form>
</body>
</html>
