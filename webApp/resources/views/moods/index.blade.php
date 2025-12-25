<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moods</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>
<body>
    <a href="/moods/create">Add Mood</a>
    @session('success')
    <strong>Success! {{ $value }}</strong>
    @endsession
    <table class="table-auto">
        <thead>
            <tr>
                <th>ID</th>
                <th>Type</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
            @forelse($moods as $mood)
            <tr>
                <td>{{ $mood->id }}</td>
                <td>{{ $mood->type }}</td>
                <td>
                    <a href="{{ route('moods.edit', $mood->id) }}">Update</a>
                    <form action="{{ route('moods.destroy', $mood->id) }}" method="POST">
                        @csrf
                        @method('DELETE')
                        <button type="submit" onclick="return confirm('Are You Sure to delete?')">Delete</button>
                    </form>
                </td>
            </tr>
            @empty
            <tr>
                <td colspan="3">No Mood Define!</td>
            </tr>
            @endforelse
        </tbody>
    </table>
</body>
</html>
